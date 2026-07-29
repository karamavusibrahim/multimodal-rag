"""Alignment-free table structure scoring.

`table_accuracy.py` scores numeric tokens as a bag: it answers "did the right
numbers survive?" and is blind to whether they landed in the right cells. A
table read into perfect numbers but scrambled rows scores 1.000 there and is
useless downstream -- you cannot cite "R&D expense, FY2025" from it, because the
association between the label and the figure is what got lost.

The obvious fix, cell-by-cell comparison, needs the predicted grid aligned to the
source grid: same row order, same column order, same handling of merged headers.
That alignment is itself a hard problem, and getting it wrong shows up as a
structure error that is really an alignment error.

So structure is scored *without* aligning anything. For every pair of numbers
appearing in both grids, ask a question that does not depend on where the pair
sits:

    are these two numbers in the same row?    (row-mate agreement)
    are these two numbers in the same column? (column-mate agreement)

Both grids answer independently, and the metric is the fraction of pairs they
answer the same way. Permuting rows, permuting columns, adding a title row, or
dropping a rule line leaves every answer unchanged; genuinely scrambling a cell
into the wrong row flips every pair it belongs to. That is the property wanted.

Reported alongside a random baseline, because row-mate agreement has a high
floor: in a tall thin table most pairs are *not* row-mates, so a predictor that
says "never" scores well. Without the baseline the number flatters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

# Cell separators. LaTeX uses & between cells and \\ between rows; the VLM's
# fallback prose format uses pipes and newlines. \\ may carry row spacing
# (\\[2pt]) whose argument must not leak into the next row's first cell.
_ROW_SPLIT_TEX = re.compile(r"\\\\(?:\[[^\]]*\])?|\\newline")
_RULES = re.compile(r"\\(?:top|mid|bottom|cmid|c|h)?(?:rule|line)\s*(?:\([^)]*\))?(?:\{[^}]*\})?")
_TEX_CMD = re.compile(r"\\[a-zA-Z]+\s*(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")
_NUM = re.compile(r"(?<![\w\\])-?\d+(?:\.\d+)?")
_TABULAR = re.compile(r"\\begin\{(?:tabular|tabularx|longtable)\}(.*?)\\end\{(?:tabular|tabularx|longtable)\}", re.S)
# Optional [t]/[b] placement before the column spec: \begin{tabular}[t]{lrr}.
_COLSPEC = re.compile(r"^(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}")
# \multicolumn{n}{spec}{content} is one *logical* cell spanning n columns. The
# generic _TEX_CMD cleanup would keep its first brace argument -- the span count
# -- as cell content, and collapsing the span to a single cell would shift every
# later column index in the row. Expand it to `content` followed by n-1 empty
# cells so column positions stay aligned with the typeset table.
_MULTICOL = re.compile(r"\\multicolumn\s*\{(\d+)\}\s*\{[^{}]*\}\s*\{([^{}]*)\}")


def _clean_cell(cell: str) -> str:
    cell = _RULES.sub(" ", cell)
    # Keep the *argument* of commands like \textbf{12.3}; drop the command name.
    cell = _TEX_CMD.sub(lambda m: m.group(1) or " ", cell)
    return cell.replace("{", " ").replace("}", " ")


def parse_latex_grid(body: str) -> list[list[str]]:
    body = _COLSPEC.sub("", body.lstrip(), count=1)
    body = _MULTICOL.sub(lambda m: m.group(2) + " & " * (int(m.group(1)) - 1), body)
    grid = []
    for raw_row in _ROW_SPLIT_TEX.split(body):
        cells = [_clean_cell(c) for c in raw_row.split("&")]
        if any(c.strip() for c in cells):
            grid.append(cells)
    return grid


def parse_predicted_grid(content: str) -> list[list[str]] | None:
    """Recover a grid from an extracted element, whichever shape it came in.

    The VLM emits tables three ways in practice: a real `tabular` environment
    embedded in a text element, pipe-delimited rows, or prose with no structure
    at all. Only the first two are gradeable; prose returns None rather than
    being scored as a one-row table, which would report a structure failure for
    something that never claimed to be a table.
    """
    m = _TABULAR.search(content)
    if m:
        return parse_latex_grid(m.group(1))

    rows = [ln for ln in content.splitlines() if ln.count("|") >= 2]
    if len(rows) >= 2:
        grid = []
        for ln in rows:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            # Markdown separator rows (---|---) carry no data.
            if all(set(c) <= set("-: ") for c in cells):
                continue
            grid.append(cells)
        if len(grid) >= 2:
            return grid
    return None


def _positions(grid: list[list[str]]) -> dict[str, tuple[int, int]]:
    """First (row, col) at which each numeric token appears.

    First occurrence, not all occurrences: a repeated value like "1" or "2024"
    has no single true position, and counting it many times would let one
    ambiguous token dominate the pair set.
    """
    pos: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            for tok in _NUM.findall(cell):
                try:
                    key = f"{float(tok):g}"
                except ValueError:
                    continue
                pos.setdefault(key, (r, c))
    return pos


@dataclass
class StructureScore:
    shared_numbers: int
    pairs: int
    row_agreement: float
    col_agreement: float
    row_baseline: float
    col_baseline: float
    source_shape: tuple[int, int]
    predicted_shape: tuple[int, int]

    @property
    def gradeable(self) -> bool:
        # Below ~6 shared numbers the pair count is too small to mean anything.
        return self.shared_numbers >= 6 and self.pairs >= 10


def score_structure(source_grid: list[list[str]],
                    predicted_grid: list[list[str]]) -> StructureScore:
    src, pred = _positions(source_grid), _positions(predicted_grid)
    shared = sorted(set(src) & set(pred))

    row_hits = col_hits = 0
    src_row_mates = src_col_mates = 0
    pairs = 0
    for a, b in combinations(shared, 2):
        pairs += 1
        s_row = src[a][0] == src[b][0]
        s_col = src[a][1] == src[b][1]
        row_hits += (s_row == (pred[a][0] == pred[b][0]))
        col_hits += (s_col == (pred[a][1] == pred[b][1]))
        src_row_mates += s_row
        src_col_mates += s_col

    def shape(g: list[list[str]]) -> tuple[int, int]:
        return (len(g), max((len(r) for r in g), default=0))

    # Baseline: the better of always-"same" and always-"different". This is what
    # a structure-blind predictor scores, and the metric is only meaningful to
    # the extent it beats it.
    def base(mates: int) -> float:
        if not pairs:
            return 0.0
        p = mates / pairs
        return max(p, 1 - p)

    return StructureScore(
        shared_numbers=len(shared),
        pairs=pairs,
        row_agreement=row_hits / pairs if pairs else 0.0,
        col_agreement=col_hits / pairs if pairs else 0.0,
        row_baseline=base(src_row_mates),
        col_baseline=base(src_col_mates),
        source_shape=shape(source_grid),
        predicted_shape=shape(predicted_grid),
    )
