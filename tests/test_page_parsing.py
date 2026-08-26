"""Regression tests for the page-element parser.

`extract_page` normalises whatever shape the vision model returns into a list of
elements. The shapes are not hypothetical -- the prompt asks for
`{"elements": [...]}`, models regularly answer with a bare list, and they
sometimes answer with `{"content": "<prose>"}` instead. That last one used to be
a silent corpus-shredder: a str is iterable, so the element loop walked the
prose one character at a time and emitted a single-character element per
character. No exception, no empty result, just a page turned into hundreds of
meaningless chunks that would then be embedded and indexed.
"""

from __future__ import annotations

from unittest.mock import patch

from mm_rag.extract.pages import extract_page

PROSE = "The page describes retrieval-augmented generation."


def run(reply: str) -> list[dict[str, str]]:
    with patch("mm_rag.extract.pages.chat", return_value=reply):
        return extract_page(b"", model="test")


def test_content_as_a_bare_string_stays_one_element():
    out = run('{"content": "%s"}' % PROSE)
    assert out == [{"kind": "text", "content": PROSE}]


def test_elements_as_a_bare_string_stays_one_element():
    out = run('{"elements": "%s"}' % PROSE)
    assert out == [{"kind": "text", "content": PROSE}]


def test_a_string_page_is_never_split_into_characters():
    # The failure mode, stated directly: one element per character.
    for reply in ('{"content": "%s"}' % PROSE, '{"elements": "%s"}' % PROSE):
        out = run(reply)
        assert len(out) < len(PROSE), "page was shredded into per-character elements"
        assert all(len(e["content"]) > 1 for e in out)


def test_the_documented_shapes_still_work():
    assert run('{"elements": [{"kind": "table", "content": "a | b"}]}') == [
        {"kind": "table", "content": "a | b"}
    ]
    assert run('[{"kind": "text", "content": "hello"}]') == [
        {"kind": "text", "content": "hello"}
    ]


def test_unknown_kinds_are_normalised_to_text():
    assert run('{"elements": [{"kind": "equation", "content": "E=mc2"}]}') == [
        {"kind": "text", "content": "E=mc2"}
    ]


def test_empty_page_returns_nothing_rather_than_a_blank_element():
    assert run('{"elements": []}') == []
    assert run("   ") == []
