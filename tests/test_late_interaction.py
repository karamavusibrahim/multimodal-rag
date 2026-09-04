"""Offline tests for the optional late-interaction scorer, on synthetic vectors."""

from __future__ import annotations

import numpy as np
import pytest

from mm_rag.retrieve.late_interaction import (
    maxsim,
    pool_patches,
    rank_pages,
    score_pages,
    single_vector_equivalent,
)


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_maxsim_matches_each_query_token_to_its_best_patch():
    q = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
    page = np.stack([unit(1, 0, 0), unit(0, 0, 1), unit(0, 1, 0)])
    assert maxsim(q, page) == pytest.approx(2.0)


def test_a_page_averaged_into_one_vector_loses_what_maxsim_keeps():
    # The point of late interaction: a page holding two distinct tables
    # answers a query about either; its mean vector answers neither well.
    q = np.stack([unit(1, 0, 0)])
    page = np.stack([unit(1, 0, 0), unit(0, 1, 0)])
    assert maxsim(q, page) == pytest.approx(1.0)
    assert single_vector_equivalent(q[0], page.mean(axis=0)) < 0.8


def test_single_vector_case_is_cosine():
    a, b = unit(1, 1, 0), unit(1, 0, 0)
    assert single_vector_equivalent(a, b) == pytest.approx(float(a @ b))


def test_rank_pages_is_best_first_and_stable():
    q = np.stack([unit(1, 0)])
    pages = [np.stack([unit(0, 1)]), np.stack([unit(1, 0)]),
             np.stack([unit(1, 0)]), np.stack([unit(1, 1)])]
    assert rank_pages(q, pages) == [1, 2, 3, 0]
    assert rank_pages(q, pages, top_k=2) == [1, 2]
    assert score_pages(q, pages).shape == (4,)


def test_pooling_divides_patch_count_and_keeps_a_partial_group():
    page = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)
    pooled = pool_patches(page, 3)
    assert pooled.shape == (3, 2)
    assert np.allclose(pooled[0], page[:3].mean(axis=0))
    assert np.allclose(pooled[2], page[6:].mean(axis=0))
    assert pool_patches(page, 1) is not None and pool_patches(page, 1).shape == page.shape
    with pytest.raises(ValueError):
        pool_patches(page, 0)


def test_shape_errors_are_loud():
    with pytest.raises(ValueError):
        maxsim(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        maxsim(np.ones(3), np.ones((2, 3)))
    assert maxsim(np.ones((0, 3)), np.ones((2, 3))) == 0.0
