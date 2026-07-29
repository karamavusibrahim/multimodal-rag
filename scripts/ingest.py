#!/usr/bin/env python
"""Extract structured elements from a PDF and build a searchable index.

    uv run python scripts/ingest.py --url https://arxiv.org/pdf/2005.11401 --max-pages 8
    uv run python scripts/ingest.py --pdf data/raw/paper.pdf

arXiv is the default corpus: open access, and dense with exactly the content
that defeats text-layer extraction -- results tables, plotted curves, equations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from mm_rag.extract.pages import extract_pdf  # noqa: E402
from mm_rag.nvidia import EMBED_NEMOTRON_3, embed  # noqa: E402


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with httpx.Client(timeout=120.0, follow_redirects=True) as c:
        r = c.get(url, headers={"User-Agent": "multimodal-rag/0.1"})
        r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="PDF URL (e.g. an arXiv /pdf/ link)")
    ap.add_argument("--pdf", type=Path, help="local PDF path")
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    if not args.url and not args.pdf:
        ap.error("pass --url or --pdf")

    if args.url:
        name = args.url.rstrip("/").split("/")[-1]
        pdf = download(args.url, Path("data/raw") / f"{name}.pdf")
    else:
        pdf = args.pdf

    print(f"extracting {pdf.name} (max {args.max_pages} pages) ...")
    elements = extract_pdf(pdf, max_pages=args.max_pages)
    if not elements:
        print("no elements extracted", file=sys.stderr)
        return 1

    from collections import Counter
    kinds = Counter(e.kind for e in elements)
    print(f"\n{len(elements)} elements: {dict(kinds)}")

    print(f"embedding via {EMBED_NEMOTRON_3.id} ...")
    vecs = embed([e.embed_text for e in elements],
                 model=EMBED_NEMOTRON_3.id, input_type="passage")
    mat = np.asarray(vecs, dtype=np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "elements.jsonl").write_text(
        "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in elements),
        encoding="utf-8")
    np.save(args.out / "vectors.npy", mat)
    (args.out / "meta.json").write_text(json.dumps(
        {"embed_model": EMBED_NEMOTRON_3.id, "n_elements": len(elements),
         "dim": int(mat.shape[1]), "kinds": dict(kinds)}, indent=2))

    print(f"index -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
