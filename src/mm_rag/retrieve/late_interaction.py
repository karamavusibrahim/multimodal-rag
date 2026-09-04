"""Optional late-interaction (MaxSim) scoring over multi-vector page embeddings.

ColPali (Faysse et al., arXiv 2407.01449, ICLR 2025) and its successors
(ColQwen2 / ColQwen2.5) retrieve document pages from their *images*: a
vision-language model emits one embedding per image patch, the query one
embedding per token, and relevance is the ColBERT-style MaxSim --

    score(q, p) = sum over query tokens t of  max over page patches j of  <q_t, p_j>

-- which lets a query term match the one patch that carries it (a table
header, a figure label) instead of being averaged away in a single page
vector. That is exactly the failure this repository's single-vector visual
arm can suffer on a page holding three tables: one vector has to stand for
all of them.

Two things this module is not, stated first:

- It is not a model. The hosted NVIDIA endpoint returns one vector per input
  and no hosted multi-vector page encoder was available when this was
  written, so no patch embeddings exist in this repository and no retrieval
  number is claimed. The scorer takes whatever `(n_tokens, dim)` /
  `(n_patches, dim)` arrays a provider produces -- `colpali-engine` locally,
  a future hosted endpoint -- and `eval/late_interaction_retrieval.py`
  scores such arrays against the same page-level gold as the single-vector
  arm, when they exist.
- It is not benchmarked here. The ViDoRe leaderboard is where these models
  are compared; this repository measures nothing about them.

What it is: the scoring, top-k, and the training-free token pooling the
ColPali paper describes (hierarchical mean pooling of neighbouring patch
vectors, cutting storage by the pool factor at a small cost in score
fidelity), in plain numpy, tested offline on synthetic arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a (n, dim) array, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def maxsim(query: np.ndarray, page: np.ndarray, *, normalize: bool = True) -> float:
    """ColBERT/ColPali late-interaction score of one page for one query.

    `query` is (n_query_tokens, dim); `page` is (n_patches, dim). With
    `normalize` the score is a sum of cosines, so it is bounded by the number
    of query tokens and comparable across pages for the same query.
    """
    q = _l2norm(query) if normalize else np.asarray(query, dtype=np.float32)
    p = _l2norm(page) if normalize else np.asarray(page, dtype=np.float32)
    if q.shape[1] != p.shape[1]:
        raise ValueError(f"dim mismatch: query {q.shape[1]} vs page {p.shape[1]}")
    if q.shape[0] == 0 or p.shape[0] == 0:
        return 0.0
    sims = q @ p.T                       # (n_tokens, n_patches)
    return float(sims.max(axis=1).sum())


def score_pages(query: np.ndarray, pages: Sequence[np.ndarray],
                *, normalize: bool = True) -> np.ndarray:
    """MaxSim of every page for one query, in `pages` order."""
    return np.asarray([maxsim(query, p, normalize=normalize) for p in pages],
                      dtype=np.float32)


def rank_pages(query: np.ndarray, pages: Sequence[np.ndarray],
               *, top_k: int | None = None) -> list[int]:
    """Page indices best-first. Ties broken by index so runs are stable."""
    scores = score_pages(query, pages)
    order = sorted(range(len(pages)), key=lambda i: (-scores[i], i))
    return order[:top_k] if top_k else order


def pool_patches(page: np.ndarray, factor: int) -> np.ndarray:
    """Training-free token pooling: mean of every `factor` consecutive patches.

    ColPali's storage cost is the patch count (1,024 vectors per page for the
    original model). Averaging neighbouring patch vectors -- the paper's
    hierarchical pooling, in its simplest sequential form -- divides that by
    `factor`. Scores change (a pooled patch is a blur of its members), so
    the trade is measured, never assumed: `eval/late_interaction_retrieval.py`
    reports the ranking with and without pooling when it has vectors to run
    on. A trailing partial group is pooled as-is rather than dropped.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    p = np.asarray(page, dtype=np.float32)
    if factor == 1 or p.shape[0] <= 1:
        return p
    groups = [p[i:i + factor].mean(axis=0) for i in range(0, p.shape[0], factor)]
    return np.stack(groups).astype(np.float32)


def single_vector_equivalent(query: np.ndarray, page: np.ndarray) -> float:
    """MaxSim with one query token and one page vector is plain cosine.

    Kept as a named function because it is the bridge to the existing
    single-vector arm: a provider that returns one vector per input plugs in
    here and scores identically to the current pipeline, which is the check
    that the scorer is not doing anything a single-vector reader would not.
    """
    return maxsim(np.asarray(query, dtype=np.float32).reshape(1, -1),
                  np.asarray(page, dtype=np.float32).reshape(1, -1))
