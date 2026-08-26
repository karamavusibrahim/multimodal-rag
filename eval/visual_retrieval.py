#!/usr/bin/env python
"""Optional visual page retrieval: find table pages without parsing anything.

The July research pass concluded the best published direction for table
retrieval -- ColPali-class visual document embeddings -- was blocked because
no such model was hosted-callable. That changed: `llama-nemotron-embed-vl-1b`
is now on the catalog and accepts page images as data URLs through the
standard /embeddings endpoint (the direction UEmbed, arXiv 2608.02583, argues
for: one embedding space for text queries and visual documents).

Experiment: embed every rendered page image once, embed a table-finding text
query per ground-truth table, rank pages by cosine. Comparison arm: the
existing parse-then-embed pipeline (VLM transcription elements + text
embeddings from `scripts/ingest.py`), scoring each page by its best element.
Queries are built from the first distinct alphabetic tokens in each LaTeX
table body. This is source-derived and intentionally easy: the tokens include
headers, labels, cell text and, in some tables, citation-key fragments.

Page-level gold comes from the manually inspected mapping in
`eval/ground_truth/2005.11401.json`, not from extraction overlap.

Optional by design: nothing in the default pipeline changes; this script
reads existing artifacts and writes only its own results file. Cost: one
image embedding per page + two query embeddings per table.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from table_accuracy import (  # noqa: E402
    extract_source_tables,
    fetch_source,
    load_page_ground_truth,
)
from mm_rag.extract.pages import render_pages  # noqa: E402
from mm_rag.nvidia import embed  # noqa: E402

VL_EMBED = "nvidia/llama-nemotron-embed-vl-1b-v2"
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?|[{}$~]")
_ALPHA = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


def table_terms(body: str, *, max_terms: int = 12) -> list[str]:
    """First distinct alphabetic terms of a LaTeX table body, in order."""
    text = _LATEX_CMD.sub(" ", body)
    seen: list[str] = []
    for cell in re.split(r"\\\\|&", text):
        for tok in _ALPHA.findall(cell):
            if tok.lower() not in (t.lower() for t in seen):
                seen.append(tok)
            if len(seen) >= max_terms:
                return seen
    return seen


def table_query(body: str) -> str:
    return "table reporting " + ", ".join(table_terms(body))


def rank_of(gold_page: int, ranked_pages: list[int]) -> int:
    return ranked_pages.index(gold_page) + 1 if gold_page in ranked_pages else 0


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-id", default="2005.11401")
    ap.add_argument("--pdf", type=Path, default=Path("data/raw/2005.11401.pdf"))
    ap.add_argument("--elements", type=Path,
                    default=Path("data/processed/elements.jsonl"))
    ap.add_argument("--vectors", type=Path,
                    default=Path("data/processed/vectors.npy"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/meta.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/visual_retrieval.json"))
    args = ap.parse_args()

    # -- gold: manually inspected source-table -> rendered page -----------
    gold_page = load_page_ground_truth(args.arxiv_id)

    # -- queries from LaTeX ground truth ----------------------------------
    blob = fetch_source(args.arxiv_id, Path("data/raw") / f"{args.arxiv_id}.tar.gz")
    tables = extract_source_tables(blob)
    # SourceTable.index is 1-based, matching crop_transcribe's table numbering.
    queries = {t.index: table_query(t.body or t.raw) for t in tables}

    # -- arm 1: visual page embeddings ------------------------------------
    pages, page_urls = [], []
    for page_no, png in render_pages(args.pdf):
        pages.append(page_no)
        page_urls.append("data:image/png;base64,"
                         + base64.b64encode(png).decode())
    print(f"embedding {len(pages)} page images with {VL_EMBED} ...")
    pvecs = np.asarray(embed(page_urls, model=VL_EMBED, input_type="passage",
                             batch_size=4), dtype=np.float32)
    pvecs /= np.linalg.norm(pvecs, axis=1, keepdims=True)

    qtexts = [queries[t] for t in sorted(queries)]
    qv_vl = np.asarray(embed(qtexts, model=VL_EMBED, input_type="query"),
                       dtype=np.float32)
    qv_vl /= np.linalg.norm(qv_vl, axis=1, keepdims=True)
    vl_sims = qv_vl @ pvecs.T  # (n_tables, n_pages)

    # -- arm 2: parse-then-embed (existing text pipeline) ------------------
    elements = [json.loads(l) for l in args.elements.read_text().splitlines() if l]
    evecs = np.load(args.vectors).astype(np.float32)
    evecs /= np.linalg.norm(evecs, axis=1, keepdims=True)
    text_model = json.loads(args.meta.read_text())["embed_model"]
    qv_tx = np.asarray(embed(qtexts, model=text_model, input_type="query"),
                       dtype=np.float32)
    qv_tx /= np.linalg.norm(qv_tx, axis=1, keepdims=True)
    esims = qv_tx @ evecs.T  # (n_tables, n_elements)
    el_pages = [e["page"] for e in elements]

    def page_ranking_text(qi: int) -> list[int]:
        best: dict[int, float] = {}
        for ei, p in enumerate(el_pages):
            s = float(esims[qi][ei])
            if s > best.get(p, -1e9):
                best[p] = s
        return sorted(best, key=best.get, reverse=True)

    # -- score both arms ---------------------------------------------------
    rows = []
    for qi, tno in enumerate(sorted(queries)):
        vl_ranked = [pages[j] for j in np.argsort(-vl_sims[qi])]
        tx_ranked = page_ranking_text(qi)
        rows.append({
            "table": tno, "gold_page": gold_page.get(tno),
            "query": queries[tno],
            "visual_rank": rank_of(gold_page.get(tno), vl_ranked),
            "text_rank": rank_of(gold_page.get(tno), tx_ranked),
            "visual_top3": vl_ranked[:3], "text_top3": tx_ranked[:3],
        })

    def metrics(key: str, subset: list[dict]) -> dict:
        ranks = [r[key] for r in subset if r[key]]
        n = len(subset)
        return {
            "recall@1": round(sum(1 for r in subset if r[key] == 1) / n, 3),
            "recall@3": round(sum(1 for r in subset if 0 < r[key] <= 3) / n, 3),
            "mrr": round(sum(1.0 / r for r in ranks) / n, 3),
        }

    result = {
        "arxiv_id": args.arxiv_id,
        "vl_model": VL_EMBED, "text_model": text_model,
        "n_tables": len(rows), "n_pages": len(pages),
        "all_tables": {"visual": metrics("visual_rank", rows),
                       "text": metrics("text_rank", rows)},
        "per_table": rows,
    }

    print(f"\nn={len(rows)} tables over {len(pages)} pages")
    for scope in ("all_tables",):
        m = result[scope]
        print(f"  {scope}: visual R@1 {m['visual']['recall@1']} "
              f"R@3 {m['visual']['recall@3']} MRR {m['visual']['mrr']}  |  "
              f"text R@1 {m['text']['recall@1']} "
              f"R@3 {m['text']['recall@3']} MRR {m['text']['mrr']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
