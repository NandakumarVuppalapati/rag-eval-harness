"""Chunk cleaned filing text into retrieval units with provenance metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    company_name: str
    sector: str
    form: str
    report_date: str
    source_url: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_pinecone_metadata(self) -> dict:
        return {
            "text": self.text,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "form": self.form,
            "report_date": self.report_date,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
        }


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_filing_text(text: str, filing_meta: dict) -> list[Chunk]:
    pieces = _splitter.split_text(text)
    ticker = filing_meta["ticker"]
    form = filing_meta["form"]
    report_date = filing_meta["report_date"]
    chunks = []
    for i, piece in enumerate(pieces):
        chunk_id = f"{ticker}_{form.replace('-', '')}_{report_date}_{i:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=piece,
                ticker=ticker,
                company_name=filing_meta["company_name"],
                sector=filing_meta["sector"],
                form=form,
                report_date=report_date,
                source_url=filing_meta["source_url"],
                chunk_index=i,
            )
        )
    return chunks
