#!/usr/bin/env python
"""Optional crop-then-transcribe pass: the second half of the routed
architecture.

`detection.py` established that the hosted page-element detector finds every
table page (4/4, zero confident false positives) while the single-prompt VLM
types 1 of 19. This script closes the loop: each detected table box is cropped
from the page image and sent to the VLM as a *dedicated* table transcription,
then scored against the same LaTeX ground truth as the baseline
(`table_accuracy.py`), so the two pipelines are directly comparable per table.

Optional by design -- the whole-page baseline in `table_accuracy.py` is
untouched, and this costs one extra VLM call per selected table box (five at
the detector's 0.5 confidence threshold for this paper).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
from PIL import Image  # noqa: E402

from mm_rag.extract.pages import VLM_MODEL, _vision_message, render_pages  # noqa: E402
from mm_rag.nvidia import chat  # noqa: E402
from table_accuracy import (  # noqa: E402
    extract_source_tables,
    fetch_source,
    load_page_ground_truth,
    numbers_in,
    score,
)

CROP_PROMPT = """This image is a cropped table from a document page.
Transcribe the table faithfully as a markdown table (pipe-delimited rows,
one header row). Reproduce every cell exactly as printed, including all
numbers, signs and decimal places. Do not summarize, do not omit rows or
columns, do not add commentary -- output only the table."""

# A crop tight to the detector box can clip row labels or the last column;
# a small margin costs nothing and protects against off-by-a-few-pixels boxes.
PAD = 0.015


def crop(png: bytes, box: dict[str, float]) -> bytes:
    img = Image.open(io.BytesIO(png))
    w, h = img.size
    x0 = max(0, int((box["x_min"] - PAD) * w))
    y0 = max(0, int((box["y_min"] - PAD) * h))
    x1 = min(w, int((box["x_max"] + PAD) * w))
    y1 = min(h, int((box["y_max"] + PAD) * h))
    out = io.BytesIO()
    img.crop((x0, y0, x1, y1)).save(out, format="PNG")
    return out.getvalue()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-id", default="2005.11401")
    ap.add_argument("--pdf", type=Path, default=Path("data/raw/2005.11401.pdf"))
    ap.add_argument("--detection", type=Path,
                    default=Path("eval/results/detection.json"))
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--reuse-results", type=Path,
                    help="rescore saved transcripts without making API calls")
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/crop_transcribe.json"))
    args = ap.parse_args()

    blob = fetch_source(args.arxiv_id, Path("data/raw") / f"{args.arxiv_id}.tar.gz")
    source_tables = extract_source_tables(blob)
    gold_pages = load_page_ground_truth(args.arxiv_id)
    baseline = {r["table"]: r for r in json.loads(
        Path("eval/results/table_accuracy.json").read_text())["per_table"]}

    if args.reuse_results:
        saved = json.loads(args.reuse_results.read_text())
        transcripts = saved["transcripts"]
        box_min_confidence = saved.get("box_min_confidence", 0.0)
    else:
        det = json.loads(args.detection.read_text())
        boxes_by_page: dict[int, list[dict[str, float]]] = {}
        for row in det["per_page"]:
            tables = (row.get("boxes") or {}).get("table") or []
            selected = [b for b in tables
                        if b.get("confidence", 0.0) >= args.min_confidence]
            if selected:
                boxes_by_page[row["page"]] = selected

        transcripts: list[dict[str, Any]] = []
        for page_no, png in render_pages(args.pdf):
            for i, box in enumerate(boxes_by_page.get(page_no, [])):
                print(f"transcribing page {page_no} box {i} "
                      f"(conf {box.get('confidence', 0):.3f}) ...")
                text = chat(VLM_MODEL, _vision_message(crop(png, box), CROP_PROMPT),
                            max_tokens=1500, temperature=0.0)
                transcripts.append({"page": page_no, "box": i,
                                    "confidence": box.get("confidence"),
                                    "text": text})
        box_min_confidence = args.min_confidence

    # Score exactly like the baseline: each source table matched to its
    # best-overlapping transcript, same thresholds implied by reporting both.
    rows = []
    for st in source_tables:
        best = {"recall": 0.0, "precision": 0.0, "overlap": 0, "page": None}
        eligible = [t for t in transcripts
                    if not gold_pages or t["page"] == gold_pages.get(st.index)]
        for t in eligible:
            r, p, o = score(st.numbers, numbers_in(t["text"]))
            if o > best["overlap"]:
                best = {"recall": r, "precision": p, "overlap": o,
                        "page": t["page"], "box": t["box"]}
        base = baseline.get(st.index, {})
        rows.append({"table": st.index, "gold_page": gold_pages.get(st.index),
                     "n_numbers": len(st.numbers), **best,
                     "baseline_recall": base.get("recall"),
                     "baseline_kind": base.get("kind")})

    print(f"\n{'table':>6} {'nums':>5} {'crop recall':>12} {'page recall':>12}")
    print("-" * 44)
    for r in rows:
        print(f"{r['table']:>6} {r['n_numbers']:>5} {r['recall']:>12.3f} "
              f"{(r['baseline_recall'] or 0):>12.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "arxiv_id": args.arxiv_id,
        "model": VLM_MODEL,
        "source_tables": len(source_tables),
        "n_transcripts": len(transcripts),
        "box_min_confidence": box_min_confidence,
        "per_table": rows,
        "transcripts": transcripts,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
