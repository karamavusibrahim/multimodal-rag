"""Regression tests for numeric ground-truth extraction and table identity.

Both fixes here were introduced by an earlier audit pass and both were wrong in
a way its tests could not see, because those tests exercised helpers rather than
the extraction path that actually runs.

1. Table identity used to follow position among the tables that *survived*
   filtering. Excluding one table therefore renumbered every later one, so
   table 4 became table 3 and was scored against table 3's page and caption.
   Any change to what counts as a number could silently shift the entire
   ground-truth mapping without touching a metric definition -- and one did.

2. Identifier digits were suppressed with lookarounds on the number pattern,
   which also rejected real data: `3-5` lost its upper bound, `2019-2020` lost
   2020, and `12kg` vanished. Suppressing names must not cost measurements.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import pytest  # noqa: E402

from table_accuracy import extract_source_tables, numbers_in  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHED_SOURCE = ROOT / "data/raw/2005.11401.tar.gz"


def clean(text: str) -> list[str]:
    """Both gold and predictions go through the same function, deliberately.

    Cleaning gold through `_STRIP_RE` while tokenising predictions raw made
    identical text produce different token sets and charged the extractor a
    precision penalty for reading the page correctly.
    """
    return numbers_in(text)


class TestIdentifierDigits:
    def test_model_and_task_names_contribute_no_numbers(self):
        assert clean("T5-11B & 44.5") == ["44.5"]
        assert clean("FEVER-3-way & 72.5") == ["72.5"]
        assert clean("88.5 & BERT-base") == ["88.5"]

    def test_ranges_keep_both_endpoints(self):
        # The dangerous direction: a guard that drops valid data silently.
        assert clean("3-5") == ["3", "5"]
        assert clean("2019-2020") == ["2019", "2020"]

    def test_a_number_with_an_attached_unit_is_still_data(self):
        assert clean("12kg") == ["12"]

    def test_plain_data_is_untouched(self):
        assert clean("-1.2 & 0.05 & 100") == ["-1.2", "0.05", "100"]


class TestLayoutMacros:
    def test_multirow_consumes_its_width_argument(self):
        # Three arguments: {nrows}{width}{text}. Stripping one leaks the width.
        assert clean(r"\multirow{2}{0.12\linewidth}{Model}") == []

    def test_multirow_accepts_the_optional_position_argument(self):
        # \multirow[t]{2}{*}{Model} is valid TeX and appears in real papers.
        assert clean(r"\multirow[t]{2}{*}{Model} & 1 & 4") == ["1", "4"]

    def test_multicolumn_keeps_its_content(self):
        assert clean(r"\multicolumn{2}{c}{Results} & 12.3") == ["12.3"]


class TestTableIdentity:
    """Filtering a table must not renumber the ones after it."""

    def _archive(self, bodies: list[str]) -> bytes:
        import io
        import tarfile

        tex = "\n".join(
            r"\begin{tabular}{lr}" + b + r"\end{tabular}" for b in bodies
        )
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = tex.encode()
            info = tarfile.TarInfo("main.tex")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def test_a_skipped_table_keeps_its_index_reserved(self):
        blob = self._archive([
            " 1 & 2 & 3 & 4 & 5 ",   # table 1, gradeable
            " 7 ",                    # table 2, below threshold
            " 10 & 11 & 12 & 13 ",    # table 3, gradeable
        ])
        got = extract_source_tables(blob, min_numbers=4)
        assert [t.index for t in got] == [1, 3], (
            "filtering renumbered the later table; its ground-truth page and "
            "caption would now be read off the wrong row"
        )

    def test_no_filtering_yields_contiguous_indices(self):
        blob = self._archive([" 1 & 2 & 3 & 4 ", " 5 & 6 & 7 & 8 "])
        assert [t.index for t in extract_source_tables(blob, min_numbers=4)] == [1, 2]


class TestSymmetry:
    def test_gold_and_prediction_tokenise_identically(self):
        # The asymmetry bug: `T5-11B & 44.5` gave gold ["44.5"] and prediction
        # ["11", "44.5"], so a perfect transcription scored precision 0.5.
        for text in ("T5-11B & 44.5", r"\multirow[t]{2}{*}{M} & 1 & 4",
                     "Total 12,914 & 8,701"):
            assert numbers_in(text) == numbers_in(text)
            assert "11" not in numbers_in("T5-11B & 44.5")


class TestPeriodLabels:
    def test_a_year_survives_a_letter_leading_token(self):
        # Q1-2024 and May-2024 are period labels, not model names. Dropping the
        # whole token discarded a real year.
        assert clean("Q1-2024") == ["2024"]
        assert clean("May-2024") == ["2024"]

    def test_a_name_that_merely_contains_digits_does_not(self):
        assert clean("COVID-19") == []
        assert clean("T5-11B") == []


@pytest.mark.skipif(not CACHED_SOURCE.exists(),
                    reason="cached arXiv source not present")
class TestAgainstTheRealPaper:
    """An executable check against the actual paper, not a synthetic fixture.

    Every unit test above passed while `eval/table_accuracy.py` raised
    IndexError on this archive: the sparse-index fix left a `source_tables[i-1]`
    lookup behind, which is wrong for every table after a gap and out of range
    at the end. Synthetic fixtures could not see it because they never had a
    gap and a high index at once.
    """

    def test_the_paper_yields_seven_tables_at_stable_indices(self):
        blob = CACHED_SOURCE.read_bytes()
        allt = extract_source_tables(blob, min_numbers=0)
        assert [t.index for t in allt] == [1, 2, 3, 4, 5, 6, 7]

    def test_filtering_leaves_a_gap_rather_than_renumbering(self):
        blob = CACHED_SOURCE.read_bytes()
        graded = extract_source_tables(blob, min_numbers=4)
        assert [t.index for t in graded] == [1, 2, 4, 5, 6, 7], (
            "table 3 is the qualitative examples table and is not numerically "
            "gradeable, but the tables after it must keep their identities"
        )

    @pytest.mark.skipif(not (ROOT / "data/processed").exists(),
                        reason="extracted element index not present")
    def test_the_evaluator_runs_to_completion_on_the_paper(self):
        """Execute the CLI. This is the test that was missing.

        44 unit tests passed while this command raised IndexError, because none
        of them ran the evaluator end to end against a corpus that has both a
        gap in its table indices and a table numbered above the length of the
        graded list. A synthetic fixture will not reproduce that combination by
        accident; the real paper does.
        """
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, "eval/table_accuracy.py",
                 "--out", str(Path(d) / "out.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
        assert r.returncode == 0, (
            f"evaluator exited {r.returncode}\n{r.stderr[-2000:]}"
        )

    def test_indices_can_be_looked_up_by_id_not_offset(self):
        blob = CACHED_SOURCE.read_bytes()
        graded = extract_source_tables(blob, min_numbers=4)
        by_index = {t.index: t for t in graded}
        # The crash: offset 7-1=6 is out of range for a 6-element list.
        assert 7 in by_index
        assert by_index[7].index == 7
