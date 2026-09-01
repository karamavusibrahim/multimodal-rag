# Audit — published numbers vs committed artifacts

A pass over what this repository claims against what its own result files and
code actually support. Every number below was recomputed from files in the
repo. No API calls were made, and no measured result was deleted to improve a
headline — where a claim had to come down, the retracted value and the reason
are kept.

## The headline correction: the eighth table was never in the paper

`tables/main_results.tex` exists in the arXiv source archive but is never
`\input` by the root document, so it does not render in the PDF. It was
counted as an 8th source table and scored as a retrieval target — against a
page a reader cannot see.

Removing it changes the published numbers, and not all in a flattering
direction:

| claim | was | is |
|---|---|---|
| source tables | 8 | **7** |
| table coverage | 2/8 = 25% | **2/7 = 28.6%** |
| crop-transcribe mean recall | 0.264 → 0.707 | **0.202 → 0.751** |
| visual retrieval R@1 | 1.000 (all 8) | **0.857 (6/7)** |
| visual retrieval MRR | 1.000 | **0.929** |
| text-arm MRR | 0.471 | **0.383–0.411** (bound) |

The visual arm loses its perfect score. That is the correct outcome: a metric
that scored a hit on an unrendered table was measuring the wrong thing. The
retracted entry is preserved in `eval/results/visual_retrieval.json` under
`retracted_unrendered_source_table`.

Separately, number matching is now constrained to the page each table actually
renders on. The per-table recall tail `0.200 0.200 0.130 0.111 0.085 0.049` was
coincidental digit overlap with prose elsewhere in the paper; under the page
constraint three of those entries are exactly `0.000`. The bimodal reading the
report argues for survives and gets sharper.

README and REPORT were publishing the pre-correction figures. They now match
the artifacts.

## Fixed in code

### A page returned as prose was shredded into one element per character

`src/mm_rag/extract/pages.py`

The prompt asks for `{"elements": [...]}`; models sometimes answer
`{"content": "<prose>"}` instead. The normaliser guarded against `dict` but not
`str` — and a `str` is iterable, so the element loop walked the prose one
character at a time and emitted a separate single-character `text` element for
every non-space character on the page. No exception, no empty result: the page
was silently turned into hundreds of meaningless chunks and then embedded and
indexed. Normalised before iteration; 6 regression tests added.

*Examined and cleared:* the `except` branch that indexes unparseable output as
raw text. `nvidia.chat()` raises `NvidiaError` on any non-200, so that path
only ever sees genuine model prose — it cannot index an HTTP error body as
document content. The fallback is a deliberate choice, not a bug.

### Two more classes of LaTeX noise were entering the numeric ground truth

`eval/table_accuracy.py`

The harness already strips layout directives, and the report documents that
fix. Two cases survived it:

- `\multirow` takes **three** argument groups — `{nrows}{width}{text}` — and
  the strip pattern removed one. `\multirow{2}{0.12\linewidth}{Model}` left
  `0.12` behind, so a *column width* was counted as a table number.
- Hyphenated identifiers were contributing their digits: `T5-11B` yielded
  `11`, `FEVER-3-way` yielded `3`. Model names and task labels, not data.

Both inflate the gold set the transcription is scored against. Fixed.

**The committed metrics predate this fix and cannot be recomputed offline** —
the gold numbers are parsed from the arXiv LaTeX at run time, and the archive
is not committed (`eval/ground_truth/2005.11401.json` holds only page and
caption). The direction of the correction is genuinely unpredictable, because
the same tokens were counted on *both* the gold and the transcription side, so
removing them shrinks numerator and denominator together. Flagged in REPORT
rather than silently restated.

## Scope, stated plainly

`src/mm_rag/` ships page rendering, page extraction and the NIM client.
`src/mm_rag/retrieve/` is an empty package; there is no `ask()`, and nothing
calls a reranker. Retrieval is *measured* in `eval/`, not served. The README
opened by describing the project as "RAG over PDF tables, charts and figures",
which oversells that by one step: this is an instrumented perception-and-
retrieval measurement harness for visually-rich pages. Cloning it reproduces
the numbers; it does not yet let you ask a question. Now said so in the
Limitations section.

## Left as-is, deliberately

- **`12,914` tokenises as `12` and `914`.** Pre-existing, and it applies
  symmetrically to gold and prediction, so it does not bias recall in a known
  direction. Changing number tokenisation would invalidate every published
  metric at once, which is not a change to make in an audit that cannot re-run
  them.
- **Every denominator here is 7 tables or 19 pages from one paper.** Noted in
  Limitations. For calibration against the field, the standard benchmark for
  this class of model is ViDoRe (V1+V2, NDCG@5), where late-interaction systems
  currently sit in the mid-80s. That figure comes from published leaderboards
  and is **not** verifiable from anything in this repository, which makes it
  the one number in this audit that was not checked from repo files. It is
  context for placing the work, not a comparison.

## Highest-value change left

Commit the arXiv source archive, or its checksum plus the parsed gold numbers.
Until then the ground truth is regenerated from a network fetch at run time,
every published metric is unreproducible offline, and corrections like the two
above can be applied but never verified.


---

## Second pass — corrections to this audit

Reviewed independently against `main`. The review found that the ground-truth
"fix" above broke the evaluator, which is worse than the bug it replaced.

### The number-regex fix deleted a table and renumbered the rest

Suppressing identifier digits via lookarounds on `_NUM_RE` cut the qualitative
"Examples from generation tasks" table from 9 numbers to 2. That put it under
the `min_numbers=4` gradeability threshold, so the parser returned **6 tables,
not 7** — and because the index was assigned *after* the filter, tables 4–7
became 3–6 and were scored against the wrong pages and captions.

Two separate defects, fixed separately:

- **Table identity now follows document position**, not position among the
  survivors. A table that is skipped keeps its index reserved. This is the real
  bug: any future change to what counts as a number could otherwise shift the
  whole ground-truth mapping without touching a metric definition.
- **Identifier digits are stripped by token, not by lookaround.** A hyphenated
  token starting with a letter (`T5-11B`, `FEVER-3-way`) is a name; one
  starting with a digit (`3-5`, `2019-2020`, `12kg`) is data. The lookaround
  version rejected all of the latter — `3-5` lost its upper bound, `12kg`
  vanished entirely. Suppressing names must not cost measurements.

`\multirow[t]{2}{*}{...}` — the optional-position form, valid TeX and present in
real papers — also still leaked its row count. Fixed.

With both applied, the paper again yields 7 tables at indices 1–7, with table 3
correctly excluded from scoring (it contains no data numbers, only layout
digits) while keeping its index. `tests/test_ground_truth_stability.py` pins
all of it, including that filtering does not renumber.

### Claims from this file that were wrong

- **"README and REPORT now match the artifacts"** — REPORT §5 still gave table
  1 as `1.000 → 1.000` where the artifacts say `0.000 → 0.926`.
- **"Cloning it reproduces the numbers"** — contradicted by this file's own
  closing section. The source archive is gitignored, query vectors are absent,
  and several measurements need hosted APIs. A clean clone reproduces none of
  them.
- **The `12,914` tokenisation is not neutral.** Splitting it into `12` and `914`
  lets a prediction of `12,999` score 0.5 against it rather than 0. It inflates
  partial matches and double-weights the cell.
- **The ViDoRe comparison is not offline-verifiable** and does not belong in an
  audit that claims every number was checked from repository files. Kept as
  context, labelled as such.


---

## Third pass

The second pass fixed table identity and broke the evaluator doing it.

### The sparse-index fix left a list-offset lookup behind

Making indices sparse (`[1, 2, 4, 5, 6, 7]`) while `main()` still did
`source_tables[r["table"] - 1]` meant every table after the gap read the wrong
source body, and table 7 raised `IndexError` outright. `eval/table_accuracy.py`
did not run at all. Lookups are now keyed by index.

**45 unit tests passed while the evaluator crashed.** That is the whole lesson
of this pass: none of them executed it. `TestAgainstTheRealPaper` now runs the
CLI against the cached paper and asserts exit 0, and it fails when the lookup
fix is reverted (verified). A synthetic fixture could not have caught this — it
needs a corpus with both a gap in its indices *and* a table numbered above the
length of the graded list, which the real paper has and a hand-written fixture
does not.

### Identifier cleaning was applied to gold only

Gold went through `_STRIP_RE`; predictions were tokenised raw. For identical
text `T5-11B & 44.5`, gold became `["44.5"]` and the prediction `["11", "44.5"]`
— charging the extractor a precision penalty for transcribing the page
correctly. Both sides now go through `numbers_in()`, which owns the cleaning.
The earlier tests only exercised the gold path, which is why this passed.

### Identifier stripping ate real period labels

`Q1-2024` and `May-2024` are letter-leading hyphenated tokens, so the whole
token was dropped along with its year. A four-digit year segment now survives;
`COVID-19` and `T5-11B` still contribute nothing.

### Also corrected

- The evaluator printed the gradeable count under a "source tables in LaTeX"
  label. It now prints rendered (7) and gradeable (6) separately, naming the
  skipped index. Conflating the two is how the docs and the code drifted apart.
- README reported 21 tests, and elsewhere 29; the suite has 45.
- The structure result was labelled table 1; the artifact says table 7.
- "Cloning it reproduces the numbers" is withdrawn. The source archive and page
  index are gitignored, query vectors were never committed, and several
  measurements need hosted APIs.
- The ViDoRe figure is now labelled as the one number here that cannot be
  checked from repository files.

## Still open after three passes

Recorded, not fixed:

- **The committed artifacts do not match their producers.** The current
  tokeniser gives different `n_numbers` than the saved run, `detection.json`
  lacks fields `detection.py` now writes, and `visual_retrieval.py` cannot
  regenerate the MRR bounds in its own artifact. Every published metric here
  predates a correction, and none can be regenerated without API access.
- `\input used` (unbraced) is not recognised, so a referenced file can be
  missed — the mirror image of the unrendered-table bug.
- A partial page map disables every unmapped table rather than constraining
  only mapped ones.
- Table matching is not one-to-one; one element can satisfy two source tables.
- `crop_transcribe` renumbers boxes after filtering, so recorded provenance
  points at the wrong rectangle.
- `detection.py` hardcodes the RAG paper's table pages.
- `normalize_number` uses `%g`, collapsing `1234567` and `1234568`.
- `12,914` still tokenises as `12` and `914`, which inflates partial matches.


---

## Fourth pass

### Two more identity/symmetry leaks, same families as before

- The computed-macro skip (`\pgfmathprintnumber`) sat before the index
  increment, so a macro table still shifted every later table's identity — the
  third instance of the same bug shape. Fixed; a synthetic end-to-end CLI test
  now runs the whole evaluator on a corpus with a macro table, a qualitative
  table and sparse graded indices, in a clean checkout with no gitignored
  inputs, and fails when the fix is reverted (verified).
- The year exception readmitted name digits: `BERT-2020` kept its 2020. A year
  now survives only in tokens whose other segments are period markers
  (Q1/H2/FY/month names).
- `grid_numbers` still tokenised raw, so a perfect `T5-11B | 44.5` grid scored
  precision 0.5 against cleaned gold. All number extraction now goes through
  one function.

### The artifacts are regenerated, and the honest direction was down

With extraction elements tracked and crop transcripts stored in the artifact,
both measurement artifacts are now produced offline by the committed code:

| | was published | regenerated |
|---|---|---|
| gradeable tables | 7 | **6** (qualitative table has no data numbers) |
| coverage | 2/7 (28.6%) | **1/6 (17%)** |
| per-table recall | 1.000 0.333 + tail | **1.000, five zeros** |
| matched precision | 0.817 | **0.632** |
| crop mean recall | 0.202 → 0.751 | **0.167 → 0.815** |
| any-correct-number | 4/7 → 7/7 | **1/6 → 5/6** |

The second "matched" table had only ever been matching its own `\multirow`
layout digits, and the recall tail was identifier noise. The distribution is
now perfectly bimodal — one table read perfectly, five not found — which makes
the report's detection-is-the-bottleneck argument stronger and its coverage
number worse. Both docs now carry the regenerated figures with the retraction
trail.

The reproducibility claim is correspondingly split, and the README now says so:
table accuracy and crop rescoring are offline-regenerable (the page index is
tracked — an earlier caveat here wrongly said it was gitignored); the visual
arm is not, and stays marked fixed-vintage.

## Still open after four passes

- The visual-retrieval artifact remains unverifiable offline and its producer
  cannot emit the published MRR bounds.
- `\input used` (unbraced) is still unrecognised; partial page maps still
  disable unmapped tables; matching is still not one-to-one; box provenance is
  still renumbered after filtering; `normalize_number` still collapses
  near-identical values; `detection.py` still hardcodes the RAG paper's pages;
  `pages.py` still indexes truncated non-JSON model output verbatim.
- `detection.json` still predates its producer's schema.
