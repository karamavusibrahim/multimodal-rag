# multimodal-rag — technical report

**Question.** Can a general-purpose vision-language model be used as the document
parser in a RAG pipeline, reading tables and charts off rendered PDF pages that
the text layer cannot represent?

**Answer.** Only partly, and the limiting factor is not the one that was
expected. Detection is where the pipeline loses almost everything: 8 tables in
the source paper, 2 recovered, 1 typed as a table. When the model does produce a
grid, the grid contains no invented numbers (precision over the grid region is
1.000 on both matched tables) — but "recovered" spans a real range, from one
character-perfect read to one partial read that kept 3 of 9 numbers.

| | |
|---|---|
| Corpus | arXiv 2005.11401 (Lewis et al., *RAG*), 19 pages |
| Extractor | `nvidia/nemotron-nano-12b-v2-vl`, one vision call per page at 150 DPI |
| Embeddings | `nvidia/nemotron-3-embed-1b` |
| Ground truth | LaTeX `tabular` environments from the arXiv e-print |
| Elements produced | 98 — 96 `text`, 1 `figure`, 1 `table` |

---

## 1. Method

### 1.1 Why the text layer is not enough

A PDF's text layer stores glyphs and positions, not relationships. A table
survives as a stream of strings whose row and column membership must be inferred
from coordinates, and a chart survives as vector paths with no data at all. The
premise here is to skip inference and render each page to an image, then ask a
VLM what is on it.

### 1.2 Ground truth without hand-labelling

Evaluating an extractor normally means transcribing pages by hand, or asking
another model to grade — which measures agreement between two models rather than
correctness.

arXiv avoids both. Every paper ships its LaTeX source, and every `tabular`
environment contains the exact numbers that were typeset into the rendered
table. The numbers therefore arrive through a channel that never touches the
image pipeline:

```
LaTeX source  ->  tabular environments  ->  numeric tokens   (ground truth)
PDF pages     ->  VLM extraction        ->  numeric tokens   (prediction)
```

This is the same move the sibling `sec-rag` project makes with XBRL, and it is
the single design decision that makes the rest of the report meaningful. No
judge model appears anywhere in the evaluation.

### 1.3 Metrics

**Content** — multiset recall and precision over numeric tokens, per source
table, against the best-overlapping extracted element. Recall is the number that
matters: a missed figure is silent data loss, whereas a spurious one is usually
visible.

**Structure** — pairwise agreement, computed without aligning the grids. For
every pair of numbers present in both the source and the prediction:

```
are these two numbers in the same row?      -> row-mate agreement
are these two numbers in the same column?   -> column-mate agreement
```

Cell-by-cell comparison would require the predicted grid aligned to the source:
same row order, same column order, same handling of merged headers. That
alignment is a hard problem in its own right, and when it fails it reports a
structure error that is really an alignment error. The pairwise formulation is
invariant to row permutation, column permutation, and inserted title rows, while
still flipping every pair that a genuinely misplaced cell participates in.

Both are numeric-cell-only. Row labels and column headers — the text that makes
a cell *interpretable* — are scored by neither, which is the most important
limitation of the whole evaluation (§5).

---

## 2. Results

### 2.1 Content

```
source tables in LaTeX:      8
tables extracted as tables:  1
matched tables:              2/8   (coverage 25%)
mean recall                  0.667
mean precision               0.817   (whole element, incl. surrounding prose)
grid precision               1.000   (grid region only, both matched tables)
```

Per-table recall, sorted — all 8 values:

```
1.000   0.333   |   0.200   0.200   0.130   0.111   0.085   0.049
```

An earlier version of this report printed only 6 of these values and called the
distribution "bimodal, with nothing in the middle". That was a curated list: the
two omitted 0.200s sit exactly in the claimed-empty middle (they fell below the
match threshold on overlap, not recall). The honest description is weaker but
still decisive: **no table cleared 0.35 except the one read essentially
perfectly, and most sit at coincidental-digit-overlap level.** The four lowest
scores are not degraded reads — they are chance overlap between a source table
and an unrelated prose paragraph.

The middle is also not quite empty on the transcription side. The one element
the model actually typed `"table"` (source table 4) is a *partial* read: 3 of 9
numbers kept, predicted shape 5×4 against a source of 13×5, precision 1.000. So
"either transcribed perfectly or never seen" overstates it — the run contains
one perfect read, one genuine partial read, and six misses. What survives
scrutiny is the ranking of causes: detection loses 6 tables outright;
transcription degraded 1 and lost 0 numbers to invention.

The distinction matters because it selects the fix. A spread clustered around
0.5 would indicate lossy transcription, and the response would be better OCR
prompting or higher render DPI. A distribution dominated by misses indicates a
detection failure, and no amount of transcription tuning addresses it.

**On the two precision numbers.** Whole-element precision charges the read for
every number on the page around the table — page numbers, inline citations,
narrative figures. The character-perfect read of table 1 scored 0.634 on that
basis, entirely from the appendix prose sharing its element. Scored against the
grid region alone, both matched tables come out at **1.000: the model invented
no numbers inside any grid it produced.** The whole-element figure is retained
because it is what a downstream consumer of the raw element would experience,
but it measures element segmentation, not transcription drift.

### 2.2 Structure

One table cleared the gradeability threshold (≥6 shared numbers, ≥10 pairs):

```
table 1:  23 shared numbers, 253 pairs
          row agreement 1.000   (baseline 0.913)
          col agreement 1.000   (baseline 0.735)
          source shape [9, 4]  ->  predicted shape [9, 4]
```

The baseline is the score of a structure-blind predictor that always gives the
majority answer. It is reported because row-mate agreement has a high floor — in
a tall thin table most pairs are *not* row-mates, so "never" scores 0.913 while
knowing nothing. Column agreement (0.735 → 1.000) carries more information, and
the exact shape match is independent corroboration.

### 2.3 The result that reframes the project

The best extraction in the entire run — the dataset-statistics table, recall
1.000, structure 1.000, shape exact — came back as valid LaTeX embedded in an
element the model typed `"text"`:

```
## I Number of instances per dataset
The number of training, development and test datapoints ... is shown in Table 7.
19
\begin{tabular}{c c c c}
Task & Train & Development & Test \\
Natural Questions & 79169 & 8758 & 3611 \\
...
```

The model **read the table perfectly and then declined to declare it a table.**

That content is retrievable — it is in the index, and a dense query will find
it — but it is not *addressable*. It cannot be routed to a table-aware prompt,
cited by cell, or rendered back into a grid, because nothing downstream knows
it is a grid. For a RAG system whose selling point is structured document
understanding, this is the difference between working and appearing to work.

The immediate consequence is that classification and transcription are separate
capabilities and should be separately prompted. They were conflated in a single
"read this page and label its elements" call, and the labelling half is the half
that failed.

---

## 3. Errors found in the evaluation itself

Most of the corrections during this work were to the measurement, not the
pipeline. The first three inflated apparent failure, which is the less common
direction and worth stating plainly.

**1. A single shared digit counted as a match.** Tables 7 and 8 both "matched"
the same element on an overlap of 1 out of 81 and 57 numbers. Coverage read
**100%** when the true figure was 25%. Fixed by requiring ≥3 overlapping numbers
and ≥20% recall before calling a table found.

**2. LaTeX layout digits were counted as data.** `\multicolumn{2}{c}{...}` and
`\cmidrule(lr){2-3}` are formatting directives; their arguments were entering the
ground-truth number set. Table 7 counted 81 "numbers" of which roughly 21 were
column spans — deflating recall for a reason unrelated to reading the page.
After stripping, it has 59.

**3. Column specifications leaked widths.** `p{0.3\textwidth}` contributed `0.3`
as a table value.

A second audit pass (2026-07-29) found four more, of which the first two changed
reported conclusions and the rest were latent:

**4. Precision was scored against the whole element, prose included.** A
character-perfect transcription scored 0.634 because the appendix page's own
prose numbers counted as "invented figures". Fixed by adding grid-region
precision (§2.1); the corrected value is 1.000 on both matched tables.

**5. The recall list was curated.** Two of the eight per-table recalls were
omitted from the published list, and they sat exactly where the "bimodal —
nothing in the middle" claim needed emptiness. §2.1 now prints all eight and
states the weaker claim the data supports.

**6. The structure ground truth was parsed from a different cleaning of the
LaTeX than the content ground truth.** `\multicolumn{2}{c}{...}` kept its span
count "2" as cell content in the structure grid (six of the eight source tables
were affected), and a body-row multicolumn shifted every later column index.
Unaffected on published numbers only because table 1 contains no multicolumn.
Fixed by expanding multicolumn spans into the correct number of cells; also
fixed: `\\[2pt]` row-spacing leakage and `\begin{tabular}[t]{...}` placement
arguments defeating column-spec stripping.

**7. Known residual holes, documented rather than fixed.** Repeated identical
numbers are keyed by first occurrence, so a grid with duplicated values is not
strictly permutation-invariant (table 1 contains three duplicated tokens and
scored 1.000 because both grids shared reading order); number normalization
collapses values beyond 6 significant digits (`%g`); Unicode minus and
scientific notation are asymmetric between the LaTeX and VLM sides. None affect
the current corpus — the corpus was checked — but any of them could bite a new
paper silently.

The first honest number reported for this project (mean recall 0.138) was worse
than the final one (0.667) because the metric was broken, not because the
pipeline improved. An evaluation harness is code, and it deserves the same
suspicion as the code it grades — the structure metric therefore ships with 21
unit tests pinning its claimed invariances (row permutation, column permutation,
title-row insertion), its claimed detections (transposition, a cell moved to
the wrong row, total flattening), and the parsing fixes above. Without them,
"row agreement 1.000" is unfalsifiable: it could be measuring row-*count*
similarity and nobody would know.

---

## 4. Engineering findings

**Prompt shape beat model choice.** The first extraction prompt led with the
output schema and a list of prohibitions ("do not invent", "do not include", "do
not guess") and returned `[]` for pages that were plainly readable. Restructuring
it task-first — describe the page, then constrain the output — took element yield
from 28 to 71 on the same 8 pages with the same model. Negative-heavy
instructions make a VLM conservative in a way that reads as incapacity.

**Accept both output shapes.** The model alternates between `{"elements": [...]}`
and a bare `[...]`. Enforcing one shape burns a retry on output that was
correct.

**Latency is highly variable and does not fail loudly.** Observed generation
times on identical call shapes: 4.2s, 22.0s, 45.0s, 50.5s. A stalled SSE stream
that trickles keepalives never trips an httpx read timeout, so a stall presents
as an indefinite hang with no error. Any batch job over this API needs a
wall-clock deadline per item, not just a socket timeout.

---

## 5. Limitations

- **Structure is measured on one table.** The metric is tested; n=1 gradeable is
  an existence proof that lossless extraction happens, not evidence about how
  often it happens.
- **One paper, 8 tables.** Enough to establish the bimodal shape and to rule out
  lossy transcription as the cause. Not enough for a stable coverage percentage.
- **Numeric cells only.** Row labels and column headers are unscored by both
  metrics. A table with the right numbers under the wrong headers scores
  perfectly here and is wrong in use. This is the next metric and it is the one
  that decides whether cell-level citation is viable.
- **Charts are unverified.** The model reads labelled values off figures, and
  nothing checks them against the underlying data. There is no arXiv-source
  equivalent for a plotted series.
- **Cost scales linearly with pages** — one vision call each, no caching across
  runs.
- **`nemoretriever-parse`, the purpose-built layout model, was not compared.**
  The general VLM was chosen on prior OCR benchmarking. Given that the failure is
  specifically *layout detection*, a layout-specialised model is the most
  promising single experiment left undone, and it is plausible that it reverses
  the headline result.
- **Matching has no one-to-one constraint and a permissive floor.** Source
  tables 1 and 2 both best-matched the same page element; table 4 — half of the
  coverage headline — passed at exactly the minimum overlap of 3, and overlap is
  unweighted, so three shared small integers suffice. Coverage is best read as
  approximate.

## 6. The detection pass — run, and it settles the question

The dedicated detector was run on 2026-07-29 (`eval/detection.py`,
`nemoretriever-page-elements-v2` hosted at `ai.api.nvidia.com/v1/cv/...`,
one base64-PNG call per page):

```
verified table pages:   6, 7, 8, 19     (by inspecting the page images)
detector table pages:   6, 7, 8, 19  + one low-confidence extra (p17)
true-table confidences: 0.858 – 0.941
false positive:         0.118  (a screenshot of a tabular-looking UI)

page-level recall:  4/4  (0 false positives above a 0.5 threshold)
VLM single-prompt:  1/19 pages typed "table"
```

Two things fell out beyond the headline:

**The detector audited the eval's own ground truth.** The reference mapping
derived from best-overlap content matching placed tables on pages 2 and 4;
the detector found nothing there, and inspection confirmed those pages hold
only prose and a figure — the "matches" were the weak coincidental-overlap
ones §5 warned about. Meanwhile page 8, which holds **three** real tables,
was absent from the derived mapping entirely. Where the detector and the
derived truth disagreed, the detector was right every time.

**Confidence separates cleanly.** Every true table scored ≥0.858; the single
false positive scored 0.118. The 0.5 threshold used in `eval/detection.py`
sits in the middle of a wide, empty gap.

Conclusion: the architecture change the content eval pointed at — detect
first with a dedicated model, then transcribe — is validated on the detection
half at a cost of one cheap CV call per page. This is NVIDIA's own nv-ingest
architecture, and it runs hosted with the same API key.

Remaining experiments, in order of leverage:

1. **Crop each detected table box and send it as a dedicated VLM table read**,
   then re-run `table_accuracy.py` against the crops — tests whether routed
   transcription recovers the tables the single-prompt read only partially
   transcribed (the boxes are already in `eval/results/detection.json`,
   normalized coordinates).
2. **`nemotron-parse` (the renamed nemoretriever-parse) as a comparison arm** —
   confirmed present in the hosted catalog; typed elements + reading order in
   one call.
3. **Header/label scoring** — the metric that decides whether cell-level
   citation is viable (§5), and the only one of these that requires new
   ground-truth parsing rather than new calls.

## 7. Conclusion

The pipeline works as a transcriber and fails as a detector — and the fix is
now measured, not merely indicated. Against LaTeX ground truth the
single-prompt VLM pass recovers 25% of the paper's tables and types 1 of 19
pages as containing one; a dedicated hosted detector finds **all four**
table-bearing pages with zero confident false positives, for one cheap CV call
per page. Of what the VLM transcribes, it invents nothing (grid precision
1.000 on both matched tables), reproduces one table exactly — values and
structure — and reads the other only partially. What remains open is the
second half of the routed architecture: transcribing from the detector's
crops rather than the whole page.

The practically useful finding is that these are two different problems that were
being solved by one prompt. The fix indicated by the data is a dedicated
page-layout pass whose only job is to decide *where* the tables are, with
transcription as a second call over each detected region — not a better
transcription prompt, which is what the 0.138 first measurement would have
suggested before the metric was repaired.
