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

Reference truth: the pages holding the 8 `tabular` environments from the
paper's LaTeX source, located via each table's best-overlap element in
`table_accuracy.py`. That mapping is exact for strong matches and approximate
for the weak ones, so per-page disagreements are reported for manual
inspection rather than silently scored.
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

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from mm_rag.extract.pages import render_pages  # noqa: E402

DETECT_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"

# Pages actually containing typeset tables, verified by inspecting the page
# images. The first version of this constant ({2, 4, 6, 7, 19}) was derived
# from table_accuracy.py's best-overlap elements -- and pages 2 and 4 turned
# out to hold no tables at all (their "matches" were the weak coincidental
# ones that audit flagged), while page 8 holds THREE tables the weak matching
# had assigned elsewhere. The detector disagreed with the derived mapping on
# exactly those pages, and the detector was right each time.
TABLE_PAGES = {6, 7, 8, 19}

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
    ap.add_argument("--pdf", type=Path, default=Path("data/raw/2005.11401.pdf"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/detection.json"))
    args = ap.parse_args()

    api_key = os.environ["NVIDIA_API_KEY"]
    rows = []
    for page_no, png in render_pages(args.pdf):
        data = detect(png, api_key)
        raw = data["data"][0].get("bounding_boxes") or {}
        boxes = {cls: [b for b in items if b.get("confidence", 0.0) >= MIN_CONF]
                 for cls, items in raw.items()}
        counts = {cls: len(items) for cls, items in boxes.items() if items}
        confs = {cls: [round(b.get("confidence", 0.0), 3) for b in items]
                 for cls, items in boxes.items() if items}
        rows.append({"page": page_no, "counts": counts, "confidences": confs,
                     "boxes": boxes})
        print(f"page {page_no:>2}: {counts or '(nothing)'}")

    detected = {r["page"] for r in rows if r["counts"].get("table")}
    tp = sorted(detected & TABLE_PAGES)
    fn = sorted(TABLE_PAGES - detected)
    fp = sorted(detected - TABLE_PAGES)

    print(f"\ntable pages (from LaTeX-source matching): {sorted(TABLE_PAGES)}")
    print(f"detector found tables on:                 {sorted(detected)}")
    print(f"  hits {tp}   missed {fn}   extra {fp}")
    print(f"page-level recall    {len(tp)}/{len(TABLE_PAGES)}")
    print(f"VLM single-prompt baseline: 1 page typed 'table' out of 19")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "detector": "nemoretriever-page-elements-v2",
        "table_pages_reference": sorted(TABLE_PAGES),
        "detected_table_pages": sorted(detected),
        "page_recall": f"{len(tp)}/{len(TABLE_PAGES)}",
        "false_positive_pages": fp,
        "per_page": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
