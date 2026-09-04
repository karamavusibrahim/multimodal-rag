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


class TestScaffoldBoundary:
    """Broken JSON is recognised by where its quoted key sits, not by whether
    the reply contains a quote and a colon."""

    def test_prose_quoting_a_label_is_indexed(self):
        for reply in (
            '[Definition] "Precision": the fraction of retrieved items that are relevant.',
            '[Draft] The label "Result": this page contains useful prose.',
        ):
            assert run(reply) == [{"kind": "text", "content": reply}], reply

    def test_short_bracket_leading_prose_is_indexed(self):
        assert run("[Draft] Results") == [{"kind": "text", "content": "[Draft] Results"}]

    def test_truncated_containers_are_still_rejected(self):
        for reply in (
            "{'elements': [{'kind': 'text', 'content': 'This page contains useful "
            "prose about retrieval and generation",
            '[{"kind": "text", "content": "This page contains useful prose about '
            'retrieval and generation and more words here',
            '{"elements": []}',
            "{'content': 'x'",
        ):
            assert run(reply) == [], reply

    def test_a_wordless_reply_is_not_content(self):
        assert run("[1]") == []
        assert run("[1] prose about things") == [
            {"kind": "text", "content": "[1] prose about things"}]


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


def test_truncated_json_is_not_indexed_as_content():
    # A reply cut off mid-structure is scaffolding, not page text.
    assert run('{"elements":[') == []
    assert run('[{"kind": "te') == []


def test_prose_that_happens_to_start_with_a_bracket_is_kept():
    # The scaffolding test must be structural, not first-character: "[Draft]"
    # leading real prose is page content.
    out = run("[Draft] This page contains useful prose about retrieval.")
    assert out == [{"kind": "text",
                    "content": "[Draft] This page contains useful prose about retrieval."}]


def test_long_truncated_json_with_wordy_content_is_still_scaffolding():
    # Word counting admitted this; the '":' key signature does not.
    out = run('{"elements":[{"kind":"text","content":"This page contains '
              'useful prose about retrieval and generation')
    assert out == []


def test_a_parseable_prefix_does_not_swallow_the_page():
    # "[1] ..." parses as the JSON list [1]; every element is discarded and
    # the page used to vanish. The salvage path keeps the prose.
    out = run("[1] A citation-style opening with real prose following it.")
    assert len(out) == 1 and out[0]["kind"] == "text"
