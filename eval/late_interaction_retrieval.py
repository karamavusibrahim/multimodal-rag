#!/usr/bin/env python
"""Optional: score multi-vector (ColPali-style) page embeddings against the
same page-level gold as the single-vector visual arm.

No model is called here and no vectors ship with the repository. The script
takes an `.npz` produced by whatever multi-vector encoder the user has --
`colpali-engine` locally, or a hosted endpoint if one appears. Plain float
arrays only (no object arrays, so the file loads without pickle):

    pages          float (n_pages, max_patches, dim), zero-padded, page 1 first
    page_lengths   int   (n_pages,)   real patch count per page
    queries        float (n_queries, max_tokens, dim), zero-padded, in
                   ascending table-index order
    query_lengths  int   (n_queries,) real token count per query
    tables         int   (n_queries,) the table index each query belongs to

It reports R@1 / R@3 / MRR with and without token pooling, alongside the
committed single-vector visual numbers for the same tables so the comparison
is on the same gold. Until such a file exists this script is an integration
point, and the README says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm_rag.retrieve.late_interaction import pool_patches, rank_pages  # noqa: E402
from table_accuracy import load_page_ground_truth  # noqa: E402
from visual_retrieval import rank_of  # noqa: E402


def metrics(ranks: list[int]) -> dict:
    n = len(ranks)
    return {
        "recall@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        "recall@3": round(sum(1 for r in ranks if 0 < r <= 3) / n, 3),
        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 3),
    }


def unpad(stack: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Split a zero-padded (n, max_len, dim) stack into n (len_i, dim) arrays."""
    if stack.ndim != 3 or len(lengths) != stack.shape[0]:
        raise SystemExit("expected a (n, max_len, dim) array with one length per row")
    return [np.asarray(stack[i, :int(n)], dtype=np.float32)
            for i, n in enumerate(lengths)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, required=True,
                    help=".npz with pages / page_lengths / queries / "
                         "query_lengths / tables (see module doc)")
    ap.add_argument("--arxiv-id", default="2005.11401")
    ap.add_argument("--pool", type=int, nargs="*", default=[1, 3, 9],
                    help="token-pooling factors to report (1 = none)")
    ap.add_argument("--baseline", type=Path,
                    default=Path("eval/results/visual_retrieval.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("eval/results/late_interaction.json"))
    args = ap.parse_args()

    # Plain numeric arrays only; object arrays would need pickle, which is
    # not accepted for a file the user may have received from elsewhere.
    data = np.load(args.vectors, allow_pickle=False)
    pages = unpad(data["pages"], data["page_lengths"])
    queries = unpad(data["queries"], data["query_lengths"])
    tables = [int(t) for t in data["tables"]]
    if len(queries) != len(tables):
        raise SystemExit("queries and tables differ in length")
    gold = load_page_ground_truth(args.arxiv_id)
    missing = [t for t in tables if t not in gold]
    if missing:
        raise SystemExit(f"no gold page for tables {missing}")

    results: dict = {"arxiv_id": args.arxiv_id, "n_pages": len(pages),
                     "n_tables": len(tables), "vectors": str(args.vectors),
                     "patches_per_page": [int(p.shape[0]) for p in pages],
                     "by_pool_factor": {}}
    for factor in args.pool:
        pooled = [pool_patches(p, factor) for p in pages]
        rows = []
        for q, t in zip(queries, tables):
            ranked = [i + 1 for i in rank_pages(q, pooled)]
            rows.append({"table": t, "gold_page": gold[t],
                         "rank": rank_of(gold[t], ranked), "top3": ranked[:3]})
        results["by_pool_factor"][str(factor)] = {
            "metrics": metrics([r["rank"] for r in rows]), "per_table": rows,
            "stored_vectors_per_page": [int(p.shape[0]) for p in pooled]}
        m = results["by_pool_factor"][str(factor)]["metrics"]
        print(f"pool x{factor}: R@1 {m['recall@1']}  R@3 {m['recall@3']}  "
              f"MRR {m['mrr']}")

    if args.baseline.exists():
        base = json.loads(args.baseline.read_text())
        results["single_vector_baseline"] = {
            "source": str(args.baseline),
            "visual": base.get("all_tables", {}).get("visual"),
        }
        print(f"single-vector visual arm (committed): "
              f"{results['single_vector_baseline']['visual']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
