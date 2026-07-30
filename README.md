# multimodal-rag

RAG over PDF **tables, charts and figures** — the content that text-layer
extraction silently drops.

Each page is rendered to an image and read by a vision model, so a table is seen
as a table and a chart as a chart. Built on NVIDIA NIM:
`nemotron-nano-12b-v2-vl` for page understanding, `nemotron-3-embed-1b` for
retrieval.

## Why not just extract the text layer

A PDF's text layer gives you characters and coordinates, not structure:

- A **table** comes out as a stream of numbers whose column association exists
  only implicitly in x-positions. Row/column meaning is gone.
- A **chart** comes out as *nothing at all* — its data lives in vector drawing
  paths, not text.
- **Equations** come out as scrambled glyph sequences.

So text-layer extraction drops precisely the content that motivates multimodal
RAG in the first place. This project pays one vision-model call per page to get
it back, and that cost is the entire point.

## Prompt design mattered more than model choice

The first version returned a bare `[]` — zero elements — for several dense
two-column pages. Not a parse failure and not a timeout: the model genuinely
answered "nothing here."

Diagnosing it was worth the detour. Asked to *describe* the same page in prose,
the model produced an accurate summary. It could read the page perfectly well.
The problem was the prompt: it led with a JSON schema and a list of prohibitions
("do not invent content that is not visible"), which the model read as licence to
bail.

Restructuring fixed it — lead with the task, put the schema last, and explicitly
forbid an empty result:

| | elements extracted | tables | figures |
|---|---|---|---|
| schema-first, prohibition-heavy | 28 | 0 | 1 |
| task-first, "never return empty" | **71** | **2** | 1 |

Same model, same pages, same 8-page PDF. **Negative-heavy instructions make
vision models conservative** — that generalizes well beyond this project.

One page still returns empty, so `extract_pdf` falls back to unstructured prose
transcription rather than leaving a hole in the corpus. Worse metadata beats
missing content.

## Pipeline

```
PDF ─▶ render page @150 DPI ─▶ VLM structured read ─▶ elements ─▶ embed ─▶ retrieve
                                     │                                        │
                                     └── fallback: prose transcription        ▼
                                                                    verify against
                                                                    original pixels
```

- **150 DPI, max 1600px edge.** The accuracy/cost knee: small table type stays
  legible while the base64 payload stays inside request limits. Prior work on
  this stack found full-2K multi-image payloads stall the free tier.
- **One image per request.** Several NVIDIA VLMs reject more than one
  (`"At most 1 image may be provided"`), so pages are sent individually rather
  than batched.
- **`pypdfium2` for rendering** — a self-contained wheel with no poppler system
  dependency, which matters for something that has to run on a laptop.
- **Verification against pixels.** `ask_page()` re-reads the *original page
  image* after retrieval identifies it, rather than trusting the extracted text.
  This catches transcription errors before they reach an answer.

## Setup

```bash
uv sync
cp .env.example .env    # NVIDIA_API_KEY=nvapi-...
```

## Usage

```bash
# arXiv is the default corpus: open access, dense with results tables and plots
uv run python scripts/ingest.py --url https://arxiv.org/pdf/2005.11401

# or a local file
uv run python scripts/ingest.py --pdf data/raw/report.pdf
```

Output (full 19-page run):

```
  page   7: 14 elements (figure, table, text)
98 elements: {'text': 96, 'figure': 1, 'table': 1}
embedding via nvidia/nemotron-3-embed-1b ...
index -> data/processed
```

That `'table': 1` is the headline result, not a formatting detail — see below.

## Measuring it: LaTeX source as ground truth

The hard part of evaluating a vision extractor is getting ground truth without
hand-transcribing pages. arXiv gives it away: every paper ships its **LaTeX
source**, and every `tabular` environment contains the exact numbers that were
typeset into the rendered table. So the truth arrives through a channel entirely
independent of the image pipeline — the same trick `sec-rag` plays with XBRL.
No hand-labelling, no LLM judge, no circularity.

```bash
uv run python eval/table_accuracy.py --arxiv-id 2005.11401
```

Measured on the full 19-page RAG paper (Lewis et al., 2020):

```
source tables in LaTeX:      8
matched tables:              2/8   (coverage 25%)
mean recall                  0.667
mean precision               0.817   (whole element, incl. surrounding prose)
grid precision               1.000   (grid region only, both matched tables)
```

**Coverage is the failure, not fidelity.** The per-table recalls, all 8:

```
1.000  0.333  |  0.200  0.200  0.130  0.111  0.085  0.049
```

No table cleared 0.35 except the one read essentially perfectly; the tail is
coincidental digit overlap with prose paragraphs, not degraded reads. (An
earlier version of this list omitted the two 0.200s and claimed a clean bimodal
split — see REPORT §3.) That still points at *page-level table detection* as
the thing to fix, and rules out "the OCR is a bit lossy", which would have
produced a spread around 0.5 and a completely different remedy. Inside any grid
the model actually produced, it invented no numbers: grid-region precision is
1.000 — the lower whole-element figure is the surrounding prose being charged
against the table.

**Structured output is the weaker half.** Of 19 pages, exactly **one** yielded an
element typed `kind: "table"`. The best result in the whole run — the dataset
statistics table, recall 1.000 — came back as valid LaTeX `\begin{tabular}`
buried inside an element typed `text`. The model *read* it perfectly and then
declined to declare it a table. That content is retrievable but not addressable:
it cannot be routed to a table-aware prompt, cited by cell, or rendered back.

Two bugs were in the harness rather than the pipeline, and both inflated failure:

- **LaTeX layout digits were counted as data.** `\multicolumn{2}{c}{...}` and
  `\cmidrule(lr){2-3}` are directives; their arguments were entering the ground
  truth. Table 7 scored 81 "numbers" of which ~21 were column spans, deflating
  recall for a reason unrelated to reading the page. Stripped, it has 59.
- **A single shared digit counted as a match.** Tables 7 and 8 both "matched" the
  same element on overlap of 1, making coverage read 100% when it was 25%. Now
  gated at ≥3 overlapping numbers and ≥20% recall.

The first honest number was worse than the second because the metric was broken,
not because the pipeline improved — worth stating, since the direction of a
correction is usually the other way.

Tables built by value-computing macros are skipped rather than counted as
failures.

## Structure, scored without aligning anything

Numeric recall says the digits survived. It cannot say they landed in the right
cells — and a table read into perfect numbers but flattened into one row scores
1.000 above while being useless, because the label-to-figure association is
exactly what makes a cell citable.

Comparing cell by cell would need the predicted grid aligned to the source grid:
same row order, same column order, same treatment of merged headers. That
alignment is its own hard problem, and when it goes wrong it reports a structure
error that is really an alignment error. So `eval/structure.py` never aligns.
For every pair of numbers present in both grids it asks two questions whose
answers don't depend on position:

```
are these two numbers in the same row?      -> row-mate agreement
are these two numbers in the same column?   -> column-mate agreement
```

Permuting rows, permuting columns, or prepending a title row leaves every answer
unchanged. Moving one value into the wrong row flips every pair it belongs to.
`tests/test_structure.py` pins both halves — the invariances and the detections —
because "row agreement 0.9" is unfalsifiable otherwise; it could just as easily
be measuring row-*count* similarity and nobody would notice.

```
$ uv run pytest tests/ -q
21 passed
```

Measured on the one gradeable table:

```
table 1:  row 1.000 (baseline 0.913)   col 1.000 (baseline 0.735)
          source shape [9, 4]  ->  predicted [9, 4]
```

The baseline is reported because row-mate agreement has a high floor — in a tall
thin table most pairs are *not* row-mates, so a predictor that always answers
"no" scores 0.913 while knowing nothing. Column agreement (0.735 → 1.000) and
the exact shape match are the load-bearing evidence.

**Together with recall 1.000, that says extraction is essentially lossless when
it fires.** The failure is entirely upstream, in deciding that a region of the
page is a table at all. This matters because it changes the fix: no amount of
better transcription prompting would have helped.

One gradeable table is thin evidence, and it is the strongest caveat on this
project — the second matched table shared only 3 numbers with its source, below
the threshold where pairwise agreement means anything.

## The fix, measured: detect, then crop, then transcribe (optional passes)

Two optional stages close the loop the baseline diagnosis opened, at one cheap
hosted call per page/box, with the whole-page baseline untouched:

**Detection** (`eval/detection.py`) — NVIDIA's hosted
`nemoretriever-page-elements-v2` detector finds **4/4 table-bearing pages with
zero confident false positives** (true tables 0.858–0.941 confidence; the one
false positive, a tabular-looking UI screenshot, at 0.118) vs the VLM's 1/19.
It also *corrected the eval's own reference mapping*: where the detector and
the LaTeX-match-derived page list disagreed, the detector was right every time.

**Crop-then-transcribe** (`eval/crop_transcribe.py`) — each detected box is
cropped and sent to the *same VLM* as a dedicated table read, scored against
the same LaTeX ground truth:

| | whole page | crops |
|---|---|---|
| mean recall (all 8 tables) | 0.264 | **0.707** |
| tables at ≥0.9 recall | 1 | **4** |
| coverage at match thresholds | 2/8 | **6/8** |

Same model, same pages: the limiting factor was never transcription ability
but attention allocation on a full page. The residual misses sit exactly where
the detector drew 2 boxes for 3 tables on one page.

## Layout

```
src/mm_rag/
  nvidia.py            NIM client: chat (incl. vision), embeddings, reranking
  extract/pages.py     render, structured page read, prose fallback, verification
eval/
  table_accuracy.py    LaTeX-source ground truth, recall/precision per table
  structure.py         alignment-free structure metric
  detection.py         optional: hosted page-element detection pass
  crop_transcribe.py   optional: dedicated VLM reads of detected table crops
scripts/
  ingest.py            PDF -> elements -> embedded index
```

## Relationship to the sibling projects

| project | scope |
|---|---|
| [`sec-rag`](../sec-rag) | hybrid retrieval + measured ablation over HTML filings |
| [`agentic-rag`](../agentic-rag) | multi-hop decomposition, self-critique, refusal |
| **`multimodal-rag`** | the visual content the other two cannot reach |

The three share an NVIDIA client and escalate in difficulty: measured retrieval →
agentic reasoning over it → perception as the input to both.

## Limitations

- **Table detection is the bottleneck: 25% coverage, 1 of 19 pages typed as a
  table.** Measured, not asserted — see above. Fidelity when a table *is* found
  is fine (recall 0.667); finding it is not.
- **Structure is measured on one table.** The metric is tested (21 unit tests
  pin its invariances), but n=1 gradeable is an existence proof that lossless
  extraction happens, not evidence about how often.
- Single paper, 8 tables. Enough to establish the bimodal shape and rule out
  lossy-OCR, not enough for a stable coverage percentage.
- Structure scoring only grades numeric cells. Row labels and column headers —
  the text that makes a cell *interpretable* — are matched by neither metric.
- One page in eight still needs the prose fallback, losing table/chart typing.
- Chart data transcription is unverified — the model reads labelled values, but
  nothing checks them against the underlying figure.
- Cost scales linearly with pages: one vision call each, no caching across runs.
- `nemoretriever-parse` is available on the catalog and is the purpose-built
  layout model; this uses the general VLM instead, which was stronger on OCR in
  prior benchmarking. Comparing the two is an obvious experiment.

## License

MIT.
