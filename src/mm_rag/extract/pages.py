"""PDF -> page images -> structured elements, via NVIDIA vision models.

**Why not just extract the text layer.** A PDF's text layer gives you characters
and coordinates, not structure. A financial table comes out as a stream of
numbers whose column association is implicit in x-positions; a chart comes out as
nothing at all, because its data lives in vector paths. Text-layer extraction
silently drops exactly the content that motivates multimodal RAG.

So each page is rendered to an image and read by a vision model, which sees the
table as a table and the chart as a chart. This costs an API call per page and is
the entire point of the project.

Two NVIDIA models do different jobs here:

  nvidia/nemoretriever-parse      document layout -> structured elements
  nvidia/nemotron-nano-12b-v2-vl  general VLM; strongest OCR of the hosted set
                                  (8/8 embedded strings at ~2.1s in prior
                                  benchmarking), used to describe charts and to
                                  answer questions against a page image

Rendering uses pypdfium2: a self-contained wheel with no poppler/system
dependency, which matters because this has to run on a laptop.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

import pypdfium2 as pdfium

from ..nvidia import chat

PARSE_MODEL = "nvidia/nemoretriever-parse"
VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

# 150 DPI is the accuracy/cost knee: small table type stays legible while the
# base64 payload stays well inside request limits. Prior VLM work on this stack
# found full-2K multi-image payloads stall the free tier.
RENDER_DPI = 150
MAX_EDGE_PX = 1600


@dataclass
class PageElement:
    doc_id: str
    page: int
    element_id: str
    kind: str          # "text" | "table" | "chart" | "figure"
    content: str       # text, serialized table, or a chart description
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def embed_text(self) -> str:
        return f"[{self.kind}] page {self.page}\n{self.content}"


def render_pages(pdf_path: Path, *, dpi: int = RENDER_DPI,
                 max_pages: int | None = None) -> Iterator[tuple[int, bytes]]:
    """Yield (page_number, PNG bytes)."""
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n):
            page = doc[i]
            scale = dpi / 72.0
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            if max(pil.size) > MAX_EDGE_PX:
                ratio = MAX_EDGE_PX / max(pil.size)
                pil = pil.resize((int(pil.width * ratio), int(pil.height * ratio)))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            yield i + 1, buf.getvalue()
    finally:
        doc.close()


def _data_url(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


def _vision_message(png: bytes, prompt: str) -> list[dict[str, Any]]:
    """OpenAI-style multimodal content array.

    Note: several NVIDIA VLMs reject more than one image per request
    ("At most 1 image may be provided"), so this deliberately sends one page at
    a time rather than batching.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(png)}},
        ],
    }]


# Prompt engineering note, learned the hard way: an earlier version led with the
# JSON schema and a list of prohibitions ("do not invent content..."). The model
# read that as licence to bail and returned a bare `[]` for dense two-column
# pages it could actually read perfectly well -- verified by asking it to
# describe the same page in prose, which it did accurately. Leading with the
# task, putting the schema last, and explicitly forbidding an empty result fixed
# it. Negative-heavy instructions make vision models conservative.
EXTRACT_PROMPT = """Read this document page and write down everything on it.

Work through the page top to bottom. For each distinct block of content, decide
what it is and transcribe it:

- Prose paragraph -> kind "text". Transcribe the text.
- Table -> kind "table". Pipe-delimited rows, header row first. Copy every
  number exactly as printed, including parentheses for negatives and any units.
- Chart or plot -> kind "chart". State what is plotted, the axis labels and
  ranges, and transcribe any labelled data values.
- Diagram, photo or equation block -> kind "figure". Describe it, including any
  labels or symbols shown.

Skip only running headers, footers and page numbers.

Every page has content. Never return an empty list -- if the page is entirely
prose, return that prose as one or more "text" elements.

Return ONLY JSON:
{"elements": [{"kind": "text", "content": "..."}]}"""


def _prose_fallback(raw: str) -> list[dict[str, str]]:
    """Index unstructurable output as prose -- if it actually is prose.

    Recognition is structural, and each earlier heuristic failed one way:
    first-character testing discarded prose that opens with a bracket
    ("[Draft] This page..."), and word-counting admitted a long truncated
    reply whose *content field* was wordy. A bracket-leading reply containing
    the JSON key signature '":' is a broken container whatever prose its
    fields carry, because intact JSON would have parsed. Anything essentially
    wordless is not page content either.
    """
    text = raw.strip()
    if not text:
        return []
    if text[0] in "{[" and re.search(r'"\s*:', text):
        return []
    if len(re.findall(r"[A-Za-z]{2,}", text[:400])) < 3:
        return []
    return [{"kind": "text", "content": text}]


def extract_page(png: bytes, *, model: str = VLM_MODEL) -> list[dict[str, str]]:
    """Extract structured elements from one rendered page."""
    raw = chat(model, _vision_message(png, EXTRACT_PROMPT),
               max_tokens=3000, temperature=0.0)
    try:
        from ..nvidia import extract_json
        data = extract_json(raw)
    except Exception:
        # A page the parser cannot structure is still worth indexing as text --
        # dropping it would silently create a hole in the corpus. Unless the
        # reply *is* a broken structure. Recognition is structural, and both
        # earlier heuristics failed one way each: first-character testing
        # discarded prose that opens with a bracket ("[Draft] This page..."),
        # and word-counting admitted a long truncated reply whose *content
        # field* was wordy. What separates them is the JSON key signature: a
        # bracket-leading reply containing '":' is a broken container whatever
        # prose its fields carry, because intact JSON would have parsed above.
        return _prose_fallback(raw)

    # Models return either {"elements": [...]} as asked, or a bare [...] list.
    # Both are unambiguous, so accept both rather than discarding good output.
    #
    # The string case is not cosmetic. A model that answers {"content": "The
    # page describes..."} puts a *str* here, and a str is iterable: the loop
    # below then walks it one character at a time and emits a separate
    # single-character "text" element for every non-space character on the
    # page. That silently shreds the page into hundreds of useless chunks
    # instead of failing, so it has to be normalised before iteration, not
    # caught inside the loop.
    if isinstance(data, list):
        items: list[Any] = data
    elif isinstance(data, dict):
        raw_items = data.get("elements") or data.get("content") or []
        if isinstance(raw_items, (dict, str)):
            items = [raw_items]
        elif isinstance(raw_items, list):
            items = raw_items
        else:
            items = []
    elif isinstance(data, str):
        items = [data]
    else:
        items = []

    out: list[dict[str, str]] = []
    for el in items:
        if not isinstance(el, dict):
            if isinstance(el, str) and el.strip():
                out.append({"kind": "text", "content": el.strip()})
            continue
        kind = str(el.get("kind", "text")).lower()
        content = str(el.get("content", "")).strip()
        if content:
            out.append({"kind": kind if kind in
                        ("text", "table", "chart", "figure") else "text",
                        "content": content})
    if not out:
        # The parse "succeeded" but yielded nothing usable -- e.g. a reply
        # opening with "[1] ..." parses as the JSON list [1], every element is
        # discarded, and the page silently vanished. An empty result from a
        # non-empty reply is a parse failure in effect, so it gets the same
        # prose salvage as an explicit one. (A genuine {"elements": []} reply
        # is caught there as a JSON container and still returns [].)
        return _prose_fallback(raw)
    return out


def extract_pdf(pdf_path: Path, *, max_pages: int | None = None,
                verbose: bool = True) -> list[PageElement]:
    """Full pipeline for one PDF."""
    doc_id = pdf_path.stem
    elements: list[PageElement] = []
    for page_no, png in render_pages(pdf_path, max_pages=max_pages):
        parsed = extract_page(png)

        # Some pages still come back empty under the structured prompt even
        # though the model reads them fine in prose (verified by asking it to
        # describe the same page). Rather than leave a hole in the corpus, fall
        # back to unstructured transcription -- worse metadata beats no content.
        if not parsed:
            prose = chat(
                VLM_MODEL,
                _vision_message(png, "Transcribe everything on this page: all "
                                     "prose, tables, and figure captions."),
                max_tokens=2500, temperature=0.0,
            ).strip()
            if prose:
                parsed = [{"kind": "text", "content": prose}]
        for j, el in enumerate(parsed):
            elements.append(PageElement(
                doc_id=doc_id,
                page=page_no,
                element_id=f"{doc_id}#p{page_no:03d}e{j:02d}",
                kind=el["kind"],
                content=el["content"],
                source_path=str(pdf_path),
            ))
        if verbose:
            kinds = ", ".join(sorted({e["kind"] for e in parsed})) or "none"
            print(f"  page {page_no:3d}: {len(parsed):2d} elements ({kinds})")
    return elements


def ask_page(png: bytes, question: str, *, model: str = VLM_MODEL) -> str:
    """Ask a question directly against a page image.

    Used as the verification step: after retrieval identifies a page, the VLM
    re-reads the original pixels rather than trusting the extracted text. This
    catches transcription errors that would otherwise propagate into the answer.
    """
    prompt = (
        f"{question}\n\n"
        "Answer using only what is visible on this page. Quote the exact figures "
        "as printed. If the page does not contain the answer, say so."
    )
    return chat(model, _vision_message(png, prompt), max_tokens=800, temperature=0.0)
