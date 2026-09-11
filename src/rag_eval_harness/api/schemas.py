"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    embedding_model: Literal["voyage", "openai"] = "voyage"
    sector_filter: str | None = None
    ticker_filter: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    score: float
    ticker: str
    company_name: str
    form: str
    report_date: str
    source_url: str
    text_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    embedding_model: str
    generation_model: str
    sources: list[SourceChunk]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    embedding_tokens: int
    generation_input_tokens: int
    generation_output_tokens: int
    estimated_cost_usd: float


class HealthResponse(BaseModel):
    status: str
    voyage_index_vectors: int
    openai_index_vectors: int
