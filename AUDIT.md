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
  currently sit in the mid-80s; the numbers here measure something narrower on
  a much smaller set and should not be read against that leaderboard.

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
