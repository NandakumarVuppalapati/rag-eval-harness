"""Retrieve chunks from Pinecone for a query, via either embedding model.

Supports both indexes built during ingestion so the same query can be
run against the domain-tuned (voyage) or general-purpose (openai)
embedding, which is what the evaluation harness needs to measure the
retrieval-quality delta between them (see docs/adr/0002).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import openai
import voyageai
from pinecone import Pinecone

EmbeddingModelName = Literal["voyage", "openai"]

_INDEX_NAMES = {
    "voyage": "rag-eval-harness-voyage",
    "openai": "rag-eval-harness-openai",
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    ticker: str
    company_name: str
    sector: str
    form: str
    report_date: str
    source_url: str


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embedding_model: EmbeddingModelName
    embedding_tokens: int
    latency_ms: float


class Retriever:
    def __init__(
        self,
        pc: Pinecone,
        voyage_client: voyageai.Client,
        openai_client: openai.OpenAI,
    ) -> None:
        self._pc = pc
        self._voyage_client = voyage_client
        self._openai_client = openai_client
        self._indexes = {name: pc.Index(idx_name) for name, idx_name in _INDEX_NAMES.items()}

    def _embed_query(self, query: str, model: EmbeddingModelName) -> tuple[list[float], int]:
        if model == "voyage":
            voyage_result = self._voyage_client.embed([query], model="voyage-finance-2", input_type="query")
            # voyageai's stubs type embeddings as list[float] | list[int] to
            # cover its optional int8 quantization mode, which this project
            # never requests (no `output_dtype` passed above) -- the float()
            # here is a no-op on the actual response, just a real runtime
            # guarantee that matches this function's declared return type
            # instead of trusting an overly-broad third-party stub.
            embedding = [float(x) for x in voyage_result.embeddings[0]]
            return embedding, voyage_result.total_tokens
        else:
            openai_result = self._openai_client.embeddings.create(model="text-embedding-3-small", input=[query])
            return openai_result.data[0].embedding, openai_result.usage.total_tokens

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        model: EmbeddingModelName = "voyage",
        sector_filter: str | None = None,
        ticker_filter: str | None = None,
    ) -> RetrievalResult:
        start = time.monotonic()
        vector, embed_tokens = self._embed_query(query, model)

        pinecone_filter = {}
        if sector_filter:
            pinecone_filter["sector"] = sector_filter
        if ticker_filter:
            pinecone_filter["ticker"] = ticker_filter

        response = self._indexes[model].query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter or None,
        )
        chunks = [
            RetrievedChunk(
                chunk_id=m["id"],
                score=m["score"],
                text=m["metadata"]["text"],
                ticker=m["metadata"]["ticker"],
                company_name=m["metadata"]["company_name"],
                sector=m["metadata"]["sector"],
                form=m["metadata"]["form"],
                report_date=m["metadata"]["report_date"],
                source_url=m["metadata"]["source_url"],
            )
            for m in response["matches"]
        ]
        latency_ms = (time.monotonic() - start) * 1000
        return RetrievalResult(
            chunks=chunks, embedding_model=model, embedding_tokens=embed_tokens, latency_ms=latency_ms
        )
