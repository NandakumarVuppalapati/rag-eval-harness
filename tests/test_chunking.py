"""Tests for chunk_filing_text() against real, already-cleaned filing text.

tests/fixtures/aapl_10q_real_narrative.txt is what extract_clean_text()
actually produces from a real 10-Q slice: long run-on paragraphs with no
"\\n\\n"/"\\n" breaks (get_text() already collapsed those), which is the
input shape this function receives in production. A hand-written multi-
paragraph stub would exercise the splitter's separators list very
differently and miss that.
"""

from __future__ import annotations

from pathlib import Path

from rag_eval_harness.ingestion.chunking import CHUNK_SIZE, chunk_filing_text

FIXTURES = Path(__file__).parent / "fixtures"

_FILING_META = {
    "ticker": "AAPL",
    "form": "10-Q",
    "report_date": "2025-12-27",
    "company_name": "Apple Inc.",
    "sector": "Technology",
    "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193",
}


def _real_narrative_text() -> str:
    return (FIXTURES / "aapl_10q_real_narrative.txt").read_text()


def test_splits_real_narrative_into_multiple_chunks():
    text = _real_narrative_text()
    chunks = chunk_filing_text(text, _FILING_META)

    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)


def test_every_chunk_is_a_verbatim_substring_of_the_source():
    # The splitter must never paraphrase, truncate mid-fact, or otherwise
    # alter the real filing text -- only cut and (for overlap) duplicate it.
    text = _real_narrative_text()
    chunks = chunk_filing_text(text, _FILING_META)

    for chunk in chunks:
        assert chunk.text in text


def test_chunk_overlap_means_combined_length_exceeds_source():
    text = _real_narrative_text()
    chunks = chunk_filing_text(text, _FILING_META)

    assert sum(len(c.text) for c in chunks) > len(text)


def test_chunk_ids_are_sequential_and_formatted_from_real_metadata():
    text = _real_narrative_text()
    chunks = chunk_filing_text(text, _FILING_META)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for i, chunk in enumerate(chunks):
        # form's "-" is stripped so chunk_id stays a single clean token,
        # matching the real 10-K/10-Q/8-K form types this project ingests.
        assert chunk.chunk_id == f"AAPL_10Q_2025-12-27_{i:04d}"


def test_short_text_produces_a_single_chunk():
    chunks = chunk_filing_text("A short real filing excerpt, well under the chunk size.", _FILING_META)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "AAPL_10Q_2025-12-27_0000"


def test_to_pinecone_metadata_carries_provenance_but_not_chunk_id():
    text = _real_narrative_text()
    chunk = chunk_filing_text(text, _FILING_META)[1]

    metadata = chunk.to_pinecone_metadata()

    assert metadata["text"] == chunk.text
    assert metadata["ticker"] == "AAPL"
    assert metadata["company_name"] == "Apple Inc."
    assert metadata["sector"] == "Technology"
    assert metadata["form"] == "10-Q"
    assert metadata["report_date"] == "2025-12-27"
    assert metadata["source_url"] == _FILING_META["source_url"]
    assert metadata["chunk_index"] == 1
    # chunk_id is the Pinecone vector ID (passed separately at upsert time),
    # not a metadata field -- to_pinecone_metadata() deliberately omits it.
    assert "chunk_id" not in metadata
