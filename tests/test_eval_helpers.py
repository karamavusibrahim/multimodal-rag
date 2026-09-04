"""Offline regressions for the eval helpers the eighth-pass reviews flagged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from table_accuracy import (  # noqa: E402
    extract_source_tables,
    normalize_number,
    numbers_in,
    page_scope,
)
from visual_retrieval import mrr_bounds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TestNormalisation:
    def test_seven_digit_figures_stay_distinct(self):
        # %g rounded both to 1.23457e+06 and a one-off transcription scored.
        assert normalize_number("1234567") != normalize_number("1234568")
        assert normalize_number("1234567") == "1234567"

    def test_formatting_still_collapses(self):
        assert normalize_number("12.30") == normalize_number("12.3") == "12.3"
        assert normalize_number("-0") == normalize_number("0") == "0"
        assert normalize_number("abc") == "abc"


class TestPeriodLabels:
    def test_calendar_units_keep_their_year(self):
        assert numbers_in("Week-2024 & 7") == ["2024", "7"]
        assert numbers_in("Month-2024 & Yr-2023") == ["2024", "2023"]

    def test_names_still_drop_their_digits(self):
        assert numbers_in("BERT-2020 & ISO-2022 & 5") == ["5"]


class TestUnbracedInput:
    def test_input_without_braces_is_followed(self):
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, text in {
                "main.tex": "\\documentclass{article}\n\\input used\n\\input{braced}\n",
                "used.tex": "\\begin{tabular}{rr} 1 & 2 \\\\ 3 & 4 \\end{tabular}",
                "braced.tex": "\\begin{tabular}{rr} 5 & 6 \\\\ 7 & 8 \\end{tabular}",
            }.items():
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        tables = extract_source_tables(buf.getvalue(), min_numbers=0)
        assert [t.numbers for t in tables] == [["1", "2", "3", "4"],
                                               ["5", "6", "7", "8"]]


class TestPartialPageMap:
    def test_unmapped_tables_are_matched_anywhere_and_marked(self):
        items = [{"page": 6, "id": "a"}, {"page": 8, "id": "b"}]
        assert page_scope({1: 6}, 1, items) == ([items[0]], True)
        assert page_scope({1: 6}, 2, items) == (items, False)
        assert page_scope({}, 2, items) == (items, False)


class TestMrrBounds:
    def test_bounds_reproduce_the_committed_text_arm_interval(self):
        art = json.loads((ROOT / "eval/results/visual_retrieval.json").read_text())
        ranks = [r["text_rank"] for r in art["per_table"]]
        assert ranks.count(None) == 1, "exactly table 5's rank is unknown"
        lo, hi = mrr_bounds(ranks, n_pages=art["n_pages"], known_beyond=3)
        assert (lo, hi) == (art["all_tables"]["text"]["mrr_lower_bound"],
                            art["all_tables"]["text"]["mrr_upper_bound"])
        assert (lo, hi) == (0.383, 0.411)

    def test_all_known_ranks_collapse_to_a_point(self):
        assert mrr_bounds([1, 2, 4], n_pages=10) == (round((1 + .5 + .25) / 3, 3),) * 2

    def test_page_count_must_exceed_the_known_top_k(self):
        with pytest.raises(ValueError):
            mrr_bounds([None], n_pages=3, known_beyond=3)
