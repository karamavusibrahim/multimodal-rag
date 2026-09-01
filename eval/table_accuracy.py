#!/usr/bin/env python
"""Measure table-extraction accuracy against arXiv LaTeX source.

The problem with evaluating a vision extractor is getting ground truth without
hand-transcribing pages. arXiv solves it: every paper ships its **LaTeX source**
alongside the PDF, and every `tabular` environment in that source contains the
exact numbers that were typeset into the rendered table.

So the numbers can be obtained through a channel completely independent of the
image pipeline -- the same trick the sibling `sec-rag` project uses with XBRL.
No hand-labelling, no LLM judge, no circularity.

    LaTeX source  ->  tabular environments  ->  numeric tokens   (ground truth)
    PDF pages     ->  VLM extraction        ->  numeric tokens   (prediction)

Metrics, per table, matched greedily to the best-overlapping extracted element:

    recall     fraction of source numbers that appear in the extraction
               -- the number that matters; misses are silent data loss
    precision  fraction of extracted numbers that exist in the source
               -- catches transcription drift and invented figures

Deliberate limitations, because they affect how the numbers should be read:

  - Only pages actually processed are eligible. A paper's later tables score
    zero recall simply for being outside `--max-pages`, so *coverage* is
    reported separately from accuracy and the headline metric is computed over
    matched tables only.
  - Numeric tokens only. Column headers, alignment and row labels are not
    scored, so a table whose numbers are right but whose structure is scrambled
    still scores well here. Structure accuracy needs a different metric.
  - LaTeX macros that compute values (\\pgfmathprintnumber, \\csname) are not
    expanded, so such tables are skipped rather than counted as failures.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure import (  # noqa: E402
    parse_latex_grid,
    parse_predicted_grid,
    score_structure,
)

ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"

_TABULAR_RE = re.compile(
    r"\\begin\{(tabular|tabularx|longtable)\}(.*?)\\end\{\1\}", re.S
)
# A "number" for our purposes: integers, decimals, percentages, with optional
# sign. Deliberately excludes bare years-in-citations by requiring it not to be
# immediately preceded by a citation command.
# A number is table *data* only if it is not part of a name. The lookbehind
# excludes `T5` (digit glued directly to a letter); hyphenated identifiers such
# as `T5-11B` and `FEVER-3-way` are removed by _IDENT_RE below instead.
#
# An earlier attempt did this with lookarounds on _NUM_RE itself. That rejected
# real data in the same stroke: `3-5` lost its upper bound, `2019-2020` lost
# 2020, and `12kg` disappeared entirely. Suppressing identifier digits must not
# cost measurements -- a ground truth that silently drops valid numbers is worse
# than one that admits a few names.
_NUM_RE = re.compile(r"(?<![\w\\])-?\d+(?:\.\d+)?")

# A hyphenated token that STARTS with a letter is a name, not a measurement:
# T5-11B, FEVER-3-way, RAG-Token, BERT-base-2. Its digits are version and task
# labels. A token starting with a digit (3-5, 2019-2020, 12kg) is data and is
# left alone.
_IDENT_RE = re.compile(r"\b[A-Za-z]\w*(?:-\w+)+\b")

# ...with one exception. `Q1-2024` and `May-2024` are letter-leading hyphenated
# tokens too, and dropping them whole discarded the year, which is a real period
# label a transcription should reproduce. A four-digit year segment survives the
# strip; everything else in the token does not, so `COVID-19` and `T5-11B` still
# contribute nothing.
_YEAR_SEGMENT = re.compile(r"^(?:19|20)\d{2}$")


# Segments that mark a token as a *period label* rather than a name: Q1-2024,
# FY-2024, H2-2024, May-2024. `BERT-2020` and `ISO-2022` are names that happen
# to end in year-shaped digits, and keeping their years reintroduced the
# identifier-digit leak this strip exists to stop.
_PERIOD_SEGMENT = re.compile(
    r"^(?:[QH][1-4]|FY|fiscal|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)$", re.I)


def _strip_identifier(m: re.Match[str]) -> str:
    segs = m.group(0).split("-")
    years = [g for g in segs if _YEAR_SEGMENT.match(g)]
    if years and all(_YEAR_SEGMENT.match(g) or _PERIOD_SEGMENT.match(g)
                     for g in segs):
        return " " + " ".join(years) + " "
    return " "

# LaTeX noise that would otherwise contribute spurious digits. The second
# alternative matters more than it looks: \multicolumn{2}{c}{...} and
# \cmidrule(lr){2-3} are *layout* directives, and their arguments were being
# counted as table content. Table 7 of the RAG paper scored 81 "numbers", ~21 of
# which were multicolumn spans -- deflating recall for a reason that has nothing
# to do with reading the page.
_STRIP_RE = re.compile(
    r"\\(?:cite|ref|label|footnote|includegraphics)\s*(?:\[[^\]]*\])?\{[^}]*\}"
    # \multirow takes THREE arguments -- {nrows}{width}{text} -- and only the
    # third is content. Stripping one group left `{0.12\linewidth}` behind, so
    # a *column width* was counted as a table number. \multicolumn is
    # {ncols}{align}{text}; its {align} carries no digits, so one group is
    # enough there, but being explicit is cheaper than rediscovering this.
    # The optional position argument is valid TeX and appears in real papers:
    # \multirow[t]{2}{*}{Model}. Without allowing for it the macro is not
    # matched at all and its row-count digit leaks into the ground truth.
    r"|\\multirow\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*\{[^}]*\}"
    r"|\\multicolumn\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*\{[^}]*\}"
    r"|\\(?:cmidrule|cline)\s*(?:\([^)]*\))?\s*\{[^}]*\}"
)
_MACRO_COMPUTED = re.compile(r"\\(pgfmathprintnumber|csname|the[a-z]+)")
# Leading column spec, allowing one level of nesting for p{...}/m{...} widths,
# and an optional [t]/[b] placement argument before it.
_COLSPEC_RE = re.compile(r"^(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}")
_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")

# Thresholds for calling a source table "found" in the extraction. Below these
# an apparent match is coincidental digit overlap rather than the table.
MIN_MATCH_OVERLAP = 3
MIN_MATCH_RECALL = 0.20


@dataclass
class SourceTable:
    index: int
    numbers: list[str]
    raw: str
    body: str = ""   # full cleaned body, kept for grid/structure scoring


def fetch_source(arxiv_id: str, cache: Path) -> bytes:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    url = ARXIV_EPRINT.format(arxiv_id=arxiv_id)
    with httpx.Client(timeout=120.0, follow_redirects=True) as c:
        r = c.get(url, headers={"User-Agent": "multimodal-rag-eval/0.1"})
        r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content


def _without_comments(tex: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def tex_files(blob: bytes) -> Iterable[str]:
    """Yield compilable TeX documents, with their inputs expanded in order.

    arXiv source archives often contain abandoned or generated ``.tex`` files
    that are not part of the rendered PDF. Scanning every member silently turns
    those files into ground truth. Prefer roots containing ``\\documentclass``
    and recursively follow only ``\\input``/``\\include`` references. A bare
    TeX file, or an archive with no identifiable root, retains the old fallback.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
            files: dict[str, str] = {}
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".tex"):
                    fh = tar.extractfile(member)
                    if fh:
                        files[member.name] = fh.read().decode("utf-8", "replace")

            roots = [name for name, text in files.items()
                     if r"\documentclass" in _without_comments(text)]
            if not roots:
                yield from files.values()
                return

            def expand(name: str, stack: tuple[str, ...] = ()) -> str:
                if name in stack or name not in files:
                    return ""
                text = _without_comments(files[name])
                parent = Path(name).parent

                def replace(match: re.Match[str]) -> str:
                    child = str(parent / match.group(1))
                    if not child.lower().endswith(".tex"):
                        child += ".tex"
                    return expand(child, stack + (name,))

                return _INPUT_RE.sub(replace, text)

            for root in roots:
                yield expand(root)
    except tarfile.ReadError:
        yield blob.decode("utf-8", "replace")


def load_page_ground_truth(arxiv_id: str, path: Path | None = None) -> dict[int, int]:
    """Load an optional, manually inspected source-table-to-PDF-page map."""
    if path is None:
        path = Path("eval/ground_truth") / f"{arxiv_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(row["table"]): int(row["page"]) for row in data["tables"]}


def normalize_number(tok: str) -> str:
    """Compare on value, not formatting: 12.30 == 12.3, -0 == 0."""
    try:
        f = float(tok)
    except ValueError:
        return tok
    return f"{f:g}"


def extract_source_tables(blob: bytes, *, min_numbers: int = 4) -> list[SourceTable]:
    tables: list[SourceTable] = []
    idx = 0
    for tex in tex_files(blob):
        for m in _TABULAR_RE.finditer(tex):
            body = m.group(2)
            if _MACRO_COMPUTED.search(body):
                # Values computed at typeset time; not recoverable here. But the
                # table still exists in the document, so it consumes an index --
                # this skip sat before the increment and shifted every later
                # table's identity, the exact bug fixed once already for the
                # gradeability filter below.
                idx += 1
                continue
            # `body` opens with the column spec -- {lrrr}, or {p{0.3\textwidth}c}
            # whose digits are widths, not data. Drop it before counting.
            body = _COLSPEC_RE.sub("", body.lstrip(), count=1)
            nums = numbers_in(body)
            # Table identity follows position in the document, NOT position among
            # the tables that survive filtering. Incrementing `idx` after the
            # filter meant that excluding one table renumbered every later one,
            # so table 4 silently became table 3 and was then scored against
            # table 3's page and caption. Any change to what counts as a number
            # -- and this file has had several -- could therefore shift the whole
            # ground-truth mapping without touching a single metric definition.
            idx += 1
            # Below the threshold the table is not numerically gradeable: the
            # qualitative "Examples from generation tasks" table in the RAG paper
            # contains no data numbers at all, only \multirow layout digits. It
            # is skipped for scoring, but it keeps its index.
            if len(nums) < min_numbers:
                continue
            tables.append(SourceTable(index=idx, numbers=nums, raw=body[:400],
                                      body=body))
    return tables


def numbers_in(text: str) -> list[str]:
    """Data numbers from any text, gold or predicted.

    Gold used to be cleaned through `_STRIP_RE` here while predictions were
    tokenised raw, so identical text produced different token sets: `T5-11B`
    contributed "11" to the prediction and nothing to the gold, charging the
    extractor a precision penalty for transcribing the page correctly. Whatever
    counts as a number has to count identically on both sides, so the cleaning
    lives inside this one function and both callers go through it.
    """
    cleaned = _IDENT_RE.sub(_strip_identifier, _STRIP_RE.sub(" ", text))
    return [normalize_number(t) for t in _NUM_RE.findall(cleaned)]


def grid_numbers(content: str) -> list[str] | None:
    """Numeric tokens from the element's isolated grid region, if it has one.

    Precision over the whole element charges the extractor for every number in
    the surrounding prose -- page numbers, inline citations, narrative figures.
    On a full-page element that made a character-perfect transcription score
    0.634. Scoring against the grid region alone measures what precision claims
    to measure: transcription drift *inside the table*.
    """
    grid = parse_predicted_grid(content)
    if grid is None:
        return None
    # Same cleaning as everywhere else. This path used raw _NUM_RE, so a grid
    # containing "T5-11B | 44.5" was charged for "11" against a gold set that
    # had stripped it -- grid precision read 0.923 on a perfect transcription.
    return [t for cell in (c for row in grid for c in row)
            for t in numbers_in(cell)]


def score(source: list[str], predicted: list[str]) -> tuple[float, float, int]:
    """Multiset recall / precision over numeric tokens."""
    from collections import Counter
    src, pred = Counter(source), Counter(predicted)
    overlap = sum((src & pred).values())
    recall = overlap / max(sum(src.values()), 1)
    precision = overlap / max(sum(pred.values()), 1)
    return recall, precision, overlap


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-id", default="2005.11401")
    ap.add_argument("--index", type=Path, default=Path("data/processed"))
    ap.add_argument("--page-ground-truth", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eval/results/table_accuracy.json"))
    args = ap.parse_args()

    elements_path = args.index / "elements.jsonl"
    if not elements_path.exists():
        print(f"no extraction at {elements_path}\n"
              f"run: uv run python scripts/ingest.py --url "
              f"https://arxiv.org/pdf/{args.arxiv_id}", file=sys.stderr)
        return 1

    elements = [json.loads(l) for l in elements_path.read_text().splitlines() if l]
    extracted_tables = [e for e in elements if e["kind"] == "table"]
    # Numbers may also land in "text" elements when the structured read falls
    # back to prose, so those count as candidate matches too -- not crediting
    # them would understate recall for a reason unrelated to reading the table.
    candidates = extracted_tables + [e for e in elements if e["kind"] == "text"]

    blob = fetch_source(args.arxiv_id, Path("data/raw") / f"{args.arxiv_id}.tar.gz")
    source_tables = extract_source_tables(blob)
    # Rendered vs gradeable are different counts and printing only the second
    # one under a "source tables" label is how the docs came to disagree with
    # the code. A table with no data numbers is still a table in the paper.
    all_source = extract_source_tables(blob, min_numbers=0)
    gold_pages = load_page_ground_truth(args.arxiv_id, args.page_ground_truth)

    skipped = sorted({t.index for t in all_source} - {t.index for t in source_tables})
    print(f"source tables rendered:      {len(all_source)}")
    print(f"  numerically gradeable:     {len(source_tables)}"
          + (f"  (skipped: {skipped}, no data numbers)" if skipped else ""))
    print(f"tables extracted from pages: {len(extracted_tables)}")
    print(f"candidate elements:          {len(candidates)}\n")

    rows: list[dict[str, Any]] = []
    for st in source_tables:
        best = {"recall": 0.0, "precision": 0.0, "overlap": 0, "element_id": None}
        best_el = None
        eligible = [el for el in candidates
                    if not gold_pages or el.get("page") == gold_pages.get(st.index)]
        for el in eligible:
            r, p, o = score(st.numbers, numbers_in(el["content"]))
            if o > best["overlap"]:
                best = {"recall": r, "precision": p, "overlap": o,
                        "element_id": el["element_id"], "kind": el["kind"]}
                best_el = el
        # Grid-scoped precision, where the element contains a parseable grid.
        # Whole-element precision stays reported for continuity, but it charges
        # the read for prose numbers that were never part of the table.
        best["precision_grid"] = None
        if best_el is not None:
            gnums = grid_numbers(best_el["content"])
            if gnums is not None:
                _, pg, _ = score(st.numbers, gnums)
                best["precision_grid"] = pg
        rows.append({"table": st.index, "gold_page": gold_pages.get(st.index),
                     "n_numbers": len(st.numbers), **best})

    # A single coincidental digit is not a match. Two tables in the first run
    # both "matched" the same element on overlap=1 of 81 and 57 numbers, which
    # made coverage read 100% when the true figure was far lower. Require a
    # substantive overlap before calling a table found.
    matched = [r for r in rows
               if r["overlap"] >= MIN_MATCH_OVERLAP and r["recall"] >= MIN_MATCH_RECALL]
    coverage = len(matched) / max(len(source_tables), 1)

    print(f"{'table':>6} {'nums':>5} {'recall':>7} {'prec':>7}  matched element")
    print("-" * 62)
    for r in rows:
        print(f"{r['table']:>6} {r['n_numbers']:>5} {r['recall']:>7.3f} "
              f"{r['precision']:>7.3f}  {r.get('element_id') or '(none)'}")

    as_table = sum(1 for r in matched if r.get("kind") == "table")
    if matched:
        mr = sum(r["recall"] for r in matched) / len(matched)
        mp = sum(r["precision"] for r in matched) / len(matched)
        with_grid = [r for r in matched if r.get("precision_grid") is not None]
        mpg = (sum(r["precision_grid"] for r in with_grid) / len(with_grid)
               if with_grid else None)
        print(f"\nmatched tables: {len(matched)}/{len(source_tables)} "
              f"(coverage {coverage:.0%}; threshold >={MIN_MATCH_OVERLAP} numbers "
              f"and >={MIN_MATCH_RECALL:.0%} recall)")
        print(f"mean recall    {mr:.3f}   (numbers present in the extraction)")
        print(f"mean precision {mp:.3f}   (whole element, incl. surrounding prose)")
        if mpg is not None:
            print(f"grid precision {mpg:.3f}   (grid region only, "
                  f"{len(with_grid)}/{len(matched)} matched tables have a grid)")
        # The distinction that matters downstream: content recovered inside a
        # prose element is retrievable but not addressable as a table, so it
        # cannot be routed, cited by cell, or rendered back.
        print(f"\nrecovered as kind='table':  {as_table}/{len(matched)}")
        print(f"recovered inside prose:     {len(matched) - as_table}/{len(matched)}")
    else:
        mr = mp = 0.0
        print("\nno table matched any extracted element")

    # --- structure ---------------------------------------------------------
    # Numeric recall says the digits survived; it cannot say they landed in the
    # right cells. A table read into perfect numbers but flattened into one row
    # scores 1.000 above and is unusable: the label-to-figure association, which
    # is the only thing that makes a cell citable, is exactly what was lost.
    by_id = {e["element_id"]: e for e in candidates}
    # Table indices are sparse: a table that is not numerically gradeable keeps
    # its index but is absent from this list, so `source_tables[table - 1]` is
    # the wrong body for every entry after the first gap -- and raises
    # IndexError once the index exceeds the list length. Look the table up by
    # the identity it actually carries.
    src_by_index = {t.index: t for t in source_tables}
    structure_rows: list[dict[str, Any]] = []
    for r in matched:
        el = by_id.get(r.get("element_id") or "")
        src = src_by_index.get(r["table"])
        if src is None:
            structure_rows.append({"table": r["table"], "gradeable": False,
                                   "why": "source table not in the graded set"})
            continue
        src_grid = parse_latex_grid(src.body)
        pred_grid = parse_predicted_grid(el["content"]) if el else None
        if pred_grid is None:
            structure_rows.append({"table": r["table"], "gradeable": False,
                                   "why": "extraction has no grid (prose)"})
            continue
        s = score_structure(src_grid, pred_grid)
        structure_rows.append({
            "table": r["table"], "gradeable": s.gradeable,
            "shared_numbers": s.shared_numbers, "pairs": s.pairs,
            "row_agreement": round(s.row_agreement, 3),
            "col_agreement": round(s.col_agreement, 3),
            "row_baseline": round(s.row_baseline, 3),
            "col_baseline": round(s.col_baseline, 3),
            "source_shape": list(s.source_shape),
            "predicted_shape": list(s.predicted_shape),
        })

    print("\n--- structure (alignment-free pairwise agreement) ---")
    for sr in structure_rows:
        if not sr["gradeable"] and "why" in sr:
            print(f"  table {sr['table']}: not gradeable -- {sr['why']}")
        else:
            flag = "" if sr["gradeable"] else "  (too few shared numbers)"
            print(f"  table {sr['table']}: row {sr['row_agreement']:.3f} "
                  f"(base {sr['row_baseline']:.3f})  col {sr['col_agreement']:.3f} "
                  f"(base {sr['col_baseline']:.3f})  "
                  f"{sr['source_shape']} -> {sr['predicted_shape']}{flag}")
    gradeable = [s for s in structure_rows if s.get("gradeable")]
    if gradeable:
        print(f"\n  mean row agreement {sum(s['row_agreement'] for s in gradeable) / len(gradeable):.3f} "
              f"vs baseline {sum(s['row_baseline'] for s in gradeable) / len(gradeable):.3f}")
        print(f"  mean col agreement {sum(s['col_agreement'] for s in gradeable) / len(gradeable):.3f} "
              f"vs baseline {sum(s['col_baseline'] for s in gradeable) / len(gradeable):.3f}")
    else:
        print("  nothing gradeable: no matched table came back with a parseable grid")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arxiv_id": args.arxiv_id,
        "source_tables": len(source_tables),
        "extracted_tables": len(extracted_tables),
        "page_ground_truth": bool(gold_pages),
        "coverage": coverage,
        "mean_recall_matched": mr,
        "mean_precision_matched": mp,
        "mean_precision_grid_matched": mpg if matched else None,
        "per_table": rows,
        "structure": structure_rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
