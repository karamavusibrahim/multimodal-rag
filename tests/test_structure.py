"""Tests for the alignment-free structure metric.

The metric's whole justification is that it measures structure without needing
the predicted grid aligned to the source. These tests pin that down: the
invariances it claims (row order, column order, added title rows) must not move
the score, and the error it claims to catch (a cell in the wrong row) must.

Without these, "row agreement 0.9" is unfalsifiable -- it could just as well be
measuring row *count* similarity and nobody would notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from structure import (  # noqa: E402
    parse_latex_grid,
    parse_predicted_grid,
    score_structure,
)

GRID = [
    ["Task", "Train", "Dev", "Test"],
    ["NQ", "79169", "8758", "3611"],
    ["TriviaQA", "78786", "8838", "11314"],
    ["WebQ", "3418", "362", "2033"],
]


class TestParsing:
    def test_latex_rows_and_cells(self):
        body = (r"{lrrr} \toprule Task & Train & Test \\ \midrule "
                r"NQ & 79169 & 3611 \\ TriviaQA & 78786 & 11314 \\ \bottomrule")
        grid = parse_latex_grid(body)
        assert len(grid) == 3
        assert [c.strip() for c in grid[1]] == ["NQ", "79169", "3611"]

    def test_colspec_digits_are_not_cells(self):
        """p{0.3\\textwidth} widths must not become row-0 data."""
        grid = parse_latex_grid(r"{p{0.3\textwidth}r} A & 12 \\ B & 34 \\")
        assert "0.3" not in " ".join(grid[0])

    def test_bold_cell_keeps_its_value(self):
        grid = parse_latex_grid(r"{cc} \textbf{44.5} & 28.9 \\ 12.0 & 9.1 \\")
        assert "44.5" in grid[0][0]

    def test_pipe_delimited_prose_table(self):
        content = "Task|Train|Test\n---|---|---\nNQ|79169|3611\nTQA|78786|11314"
        grid = parse_predicted_grid(content)
        assert grid is not None and len(grid) == 3   # separator row dropped

    def test_embedded_tabular_is_found_inside_prose(self):
        """The real case: the VLM's best read came back typed 'text'."""
        content = ("The number of datapoints is shown in Table 7.\n\n"
                   r"\begin{tabular}{c c}" "\nA & 79169 \\\\\nB & 8758 \\\\\n"
                   r"\end{tabular}")
        grid = parse_predicted_grid(content)
        assert grid is not None
        assert any("79169" in " ".join(r) for r in grid)

    def test_prose_is_not_gradeable(self):
        assert parse_predicted_grid("Revenue rose to 391.0 billion in 2024.") is None


class TestInvariance:
    def test_identical_grid_scores_perfectly(self):
        s = score_structure(GRID, GRID)
        assert s.row_agreement == 1.0 and s.col_agreement == 1.0

    def test_row_order_does_not_matter(self):
        shuffled = [GRID[0], GRID[3], GRID[1], GRID[2]]
        s = score_structure(GRID, shuffled)
        assert s.row_agreement == 1.0, "row-mate pairs are order-independent"

    def test_column_order_does_not_matter(self):
        swapped = [[r[0], r[3], r[2], r[1]] for r in GRID]
        s = score_structure(GRID, swapped)
        assert s.col_agreement == 1.0

    def test_extra_title_row_does_not_matter(self):
        s = score_structure(GRID, [["Table 7: datapoints"]] + GRID)
        assert s.row_agreement == 1.0


class TestDetection:
    def test_transposed_table_is_caught(self):
        transposed = [list(col) for col in zip(*GRID)]
        s = score_structure(GRID, transposed)
        assert s.row_agreement < 0.6, "rows became columns; must not score well"

    def test_one_value_in_the_wrong_row_costs_something(self):
        broken = [list(r) for r in GRID]
        broken[1][2], broken[2][2] = broken[2][2], broken[1][2]
        s = score_structure(GRID, broken)
        assert s.row_agreement < 1.0

    def test_flattened_to_one_row_scores_at_baseline(self):
        """The prose-fallback failure: every number correct, all structure gone."""
        flat = [[c for row in GRID for c in row]]
        s = score_structure(GRID, flat)
        # Everything is a row-mate, so agreement collapses to the share of
        # source pairs that genuinely were row-mates.
        assert s.row_agreement < 0.5


class TestBaseline:
    def test_baseline_is_reported_and_beatable(self):
        s = score_structure(GRID, GRID)
        assert s.row_agreement > s.row_baseline

    def test_baseline_exposes_a_flattering_score(self):
        """A tall thin table makes 'never a row-mate' look good; the baseline
        is what stops that from being read as success."""
        tall = [[str(i * 10 + 1), str(i * 10 + 2)] for i in range(12)]
        flat = [[c for row in tall for c in row]]
        s = score_structure(tall, [[c] for row in tall for c in row])
        assert s.row_baseline > 0.8
        assert s.row_agreement <= s.row_baseline + 1e-9
        assert score_structure(tall, flat).row_agreement < s.row_baseline

    def test_too_few_shared_numbers_is_not_gradeable(self):
        tiny = [["A", "1"], ["B", "2"]]
        assert not score_structure(tiny, tiny).gradeable


@pytest.mark.parametrize("n", [2, 3])
def test_shapes_are_reported(n):
    s = score_structure(GRID, GRID[:n])
    assert s.source_shape == (4, 4) and s.predicted_shape[0] == n


class TestLatexParsingHardening:
    """Regressions for parsing holes found in the 2026-07-29 audit. All were
    latent (absent from the current paper's source) but each silently corrupts
    the ground-truth grid when present."""

    def test_multicolumn_span_preserves_column_indices(self):
        # \multicolumn{2}{c}{Header} occupies TWO columns; collapsing it to one
        # cell would shift every later column, and keeping its first brace
        # argument would inject the span count "2" as data.
        body = (r"{lrr} \multicolumn{2}{c}{Split} & Test \\ "
                r"NQ & 79169 & 3611 \\ WebQ & 3418 & 2033 \\")
        grid = parse_latex_grid(body)
        assert len(grid[0]) == 3
        assert "2" not in " ".join(grid[0])
        # 3611 must sit in the same column as 2033, not shifted.
        from structure import _positions
        pos = _positions(grid)
        assert pos["3611"][1] == pos["2033"][1]

    def test_row_spacing_argument_does_not_leak(self):
        body = r"{lr} a & 1 \\[2pt] b & 2 \\"
        grid = parse_latex_grid(body)
        from structure import _positions
        assert "2" in _positions(grid)
        # The [2pt] spacing must not put a "2" token in row 1's first cell.
        assert not any("2pt" in c for row in grid for c in row)
        assert grid[1][0].strip() == "b"

    def test_placement_argument_before_colspec(self):
        # \begin{tabular}[t]{lrr} -- the [t] must not defeat colspec stripping,
        # which would leak p{0.3\textwidth} digits into the first row.
        body = r"[t]{p{0.3\textwidth}rr} a & 1 & 2 \\ b & 3 & 4 \\"
        grid = parse_latex_grid(body)
        from structure import _positions
        assert "0.3" not in _positions(grid)
