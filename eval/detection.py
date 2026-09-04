#!/usr/bin/env python
"""Page-layout detection pass: does a dedicated detector find the tables the
VLM read but declined to label?

The content eval (`table_accuracy.py`) established that detection, not
transcription, is where the pipeline loses tables: 1 of 19 pages produced an
element typed "table", while the best transcription in the run arrived as
valid LaTeX inside an element typed "text". This script tests the fix that
finding indicates -- a separate detection stage whose only job is to say
*where* the tables are.

Detector: `nemoretriever-page-elements-v2`, NVIDIA's hosted YOLOX page-element
model (classes: table / chart / infographic / title). It lives on the CV host
(`ai.api.nvidia.com/v1/cv/...`), takes a base64 PNG per request, and returns
bounding boxes with confidences. Like the reranker, it never appears in
`/v1/models` -- absence from the catalog list does not mean absence from the
service.

Reference truth: the pages holding the seven rendered tables, recorded in
`eval/ground_truth/2005.11401.json` after inspecting table captions and pages.
An earlier evaluator incorrectly counted an unreferenced eighth `tabular`
environment that exists in the source archive but not in the compiled PDF.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from mm_rag.extract.pages import render_pages  # noqa: E402
from table_accuracy import load_page_ground_truth  # noqa: E402

DETECT_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"

DEFAULT_ARXIV_ID = "2005.11401"
DEFAULT_PDF = Path("data/raw") / f"{DEFAULT_ARXIV_ID}.pdf"

# Pages actually containing typeset tables in the RAG paper, verified by
# inspecting the page images. The first version of this constant
# ({2, 4, 6, 7, 19}) was derived from table_accuracy.py's best-overlap
# elements -- and pages 2 and 4 turned out to hold no tables at all (their
# "matches" were the weak coincidental ones that audit flagged), while page 8
# holds THREE tables the weak matching had assigned elsewhere. The detector
# disagreed with the derived mapping on exactly those pages, and the detector
# was right each time. This is the reference for ONE paper; `table_pages()`
# below refuses to score another PDF against it.
TABLE_PAGES = {6, 7, 8, 19}


def table_pages(pdf: Path, arxiv_id: str, explicit: list[int] | None) -> set[int]:
    """The reference table pages for `pdf`.

    Priority: pages given on the command line; the manually inspected page
    map in `eval/ground_truth/<arxiv_id>.json`; the RAG-paper constant, only
    for the RAG paper. `--pdf other.pdf` used to be scored against the RAG
    paper's pages and report a recall that meant nothing.
    """
    if explicit:
        return set(explicit)
    mapped = set(load_page_ground_truth(arxiv_id).values())
    if mapped:
        return mapped
    if Path(pdf).name == DEFAULT_PDF.name:
        return set(TABLE_PAGES)
    raise SystemExit(
        f"no reference table pages for {pdf}: pass --table-pages, or add "
        f"eval/ground_truth/{arxiv_id}.json")

# Measured separation on this paper: every true table box scored >=0.858, the
# one false positive (a screenshot of a tabular-looking UI on page 17) scored
# 0.118. The threshold sits in the wide gap between those clusters.
MIN_CONF = 0.5


def detect(png: bytes, api_key: str) -> dict:
    r = httpx.post(
        DETECT_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"input": [{"type": "image_url",
                         "url": "data:image/png;base64,"
                                + base64.b64encode(png).decode()}]},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ap.add_argument("--arxiv-id", default=DEFAULT_ARXIV_ID,
                    help="selects eval/ground_truth/<id>.json as the page reference")
    ap.add_argument("--table-pages", type=int, nargs="*",
                    help="reference table pages (overrides the ground-truth map)")
    ap.add_argument("--out", type=Path, default=Path("eval/results/detection.json"))
    args = ap.parse_args()

    reference = table_pages(args.pdf, args.arxiv_id, args.table_pages)
    api_key = os.environ["NVIDIA_API_KEY"]
    rows = []
    for page_no, png in render_pages(args.pdf):
        data = detect(png, api_key)
        raw = data["data"][0].get("bounding_boxes") or {}
        boxes = {cls: items for cls, items in raw.items() if items}
        counts = {cls: len(items) for cls, items in boxes.items()}
        confs = {cls: [round(b.get("confidence", 0.0), 3) for b in items]
                 for cls, items in boxes.items()}
        confident_boxes = {
            cls: [b for b in items if b.get("confidence", 0.0) >= MIN_CONF]
            for cls, items in boxes.items()
        }
        confident_counts = {cls: len(items) for cls, items in confident_boxes.items()
                            if items}
        rows.append({"page": page_no, "counts": counts, "confidences": confs,
                     "boxes": boxes, "confident_counts": confident_counts,
                     "confident_boxes": confident_boxes})
        print(f"page {page_no:>2}: {confident_counts or '(nothing)'}")

    raw_detected = {r["page"] for r in rows if r["counts"].get("table")}
    detected = {r["page"] for r in rows if r["confident_counts"].get("table")}
    tp = sorted(detected & reference)
    fn = sorted(reference - detected)
    fp = sorted(detected - reference)

    print(f"\nreference table pages:                    {sorted(reference)}")
    print(f"detector found tables on:                 {sorted(detected)}")
    print(f"  hits {tp}   missed {fn}   extra {fp}")
    print(f"page-level recall    {len(tp)}/{len(reference)}")
    if args.pdf == DEFAULT_PDF:
        print("VLM single-prompt baseline on this paper: 1 page typed 'table' "
              "out of 19")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "detector": "nemoretriever-page-elements-v2",
        "pdf": str(args.pdf),
        "confidence_threshold": MIN_CONF,
        "table_pages_reference": sorted(reference),
        "table_pages_reference_source": (
            "command line" if args.table_pages else
            f"eval/ground_truth/{args.arxiv_id}.json"
            if load_page_ground_truth(args.arxiv_id) else "TABLE_PAGES constant"),
        "detected_table_pages": sorted(detected),
        "raw_detected_table_pages": sorted(raw_detected),
        "page_recall": f"{len(tp)}/{len(reference)}",
        "false_positive_pages": fp,
        "low_confidence_extra_pages": sorted(raw_detected - detected),
        "per_page": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
