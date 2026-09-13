"""Tests for extract_clean_text() against a real SEC filing excerpt.

tests/fixtures/aapl_10q_excerpt.htm is a trimmed slice of Apple's actual
Q1 FY2026 10-Q (see tests/fixtures/README.md for exactly how it was cut),
not synthetic markup -- the whole point of this module is stripping the
iXBRL hidden block real EDGAR filings ship with, and a hand-written stub
wouldn't exercise the real tag shapes (namespaced `ix:header`/`ix:hidden`,
a `display:none` wrapper div around them, visible narrative sitting inside
its own `ix:nonNumeric` tag) that make that stripping non-trivial. The
script/style/`.hidden`-class cases below don't need a real filing -- they're
generic HTML behavior -- so those use small literal fixtures instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_harness.ingestion.parsing import extract_clean_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_strips_ixbrl_hidden_facts_from_real_filing():
    text = extract_clean_text(FIXTURES / "aapl_10q_excerpt.htm")

    # Real XBRL facts from the filing's <ix:hidden> block -- none of this
    # is meant to be human-readable and none of it should survive.
    assert "AmendmentFlag" not in text
    assert "DocumentFiscalYearFocus" not in text
    assert "0000320193" not in text  # Apple's real EDGAR CIK
    assert "P406D" not in text  # a real XBRL duration value from the block
    assert "ix:" not in text
    assert "contextRef" not in text


def test_preserves_real_narrative_text_from_the_same_filing():
    text = extract_clean_text(FIXTURES / "aapl_10q_excerpt.htm")

    # The filing's real, visible "Basis of Presentation and Preparation"
    # note, tagged with a *visible* ix:nonNumeric (not ix:hidden) -- this
    # is exactly the content the corpus is built to keep.
    assert "Basis of Presentation and Preparation" in text
    assert (
        "condensed consolidated financial statements include the accounts "
        "of Apple Inc." in text
    )


def test_collapses_whitespace_between_real_blocks():
    text = extract_clean_text(FIXTURES / "aapl_10q_excerpt.htm")

    assert "  " not in text
    assert "\n" not in text
    assert "\t" not in text
    assert text == text.strip()


def test_strips_script_and_style_tags(tmp_path: Path):
    html_path = tmp_path / "stub.htm"
    html_path.write_text(
        "<html><head><style>body { color: red; }</style></head>"
        "<body><script>alert('should not appear');</script>"
        "<p>Real visible text.</p></body></html>"
    )

    text = extract_clean_text(html_path)

    assert text == "Real visible text."


def test_strips_inline_display_none_and_hidden_class(tmp_path: Path):
    html_path = tmp_path / "stub.htm"
    html_path.write_text(
        "<html><body>"
        '<div style="display:none">inline-hidden text</div>'
        '<div style="display: none">inline-hidden with space</div>'
        '<span class="hidden">class-hidden text</span>'
        "<p>Visible paragraph.</p>"
        "</body></html>"
    )

    text = extract_clean_text(html_path)

    assert text == "Visible paragraph."


def test_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.htm"
    with pytest.raises(FileNotFoundError):
        extract_clean_text(missing)
