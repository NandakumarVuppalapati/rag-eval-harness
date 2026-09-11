"""FastAPI service: the RAG pipeline this project evaluates.

    uvicorn rag_eval_harness.api.main:app --reload
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import anthropic
import openai
import structlog
import voyageai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pinecone import Pinecone

from rag_eval_harness.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from rag_eval_harness.generation.generator import GENERATION_MODEL, Generator
from rag_eval_harness.observability.pricing import estimate_cost_usd
from rag_eval_harness.retrieval.retriever import Retriever

load_dotenv()
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    app.state.pc = pc
    app.state.retriever = Retriever(pc, voyage_client, openai_client)
    app.state.generator = Generator(anthropic_client)
    log.info("rag_eval_harness_api_started")
    yield


app = FastAPI(
    title="RAG Evaluation & Observability Harness API",
    description="Retrieval + generation service over a real SEC-filing corpus.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pc = app.state.pc
    voyage_stats = pc.Index("rag-eval-harness-voyage").describe_index_stats()
    openai_stats = pc.Index("rag-eval-harness-openai").describe_index_stats()
    return HealthResponse(
        status="ok",
        voyage_index_vectors=voyage_stats.get("total_vector_count", 0),
        openai_index_vectors=openai_stats.get("total_vector_count", 0),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    retriever: Retriever = app.state.retriever
    generator: Generator = app.state.generator

    start = time.monotonic()
    try:
        retrieval = retriever.retrieve(
            req.question,
            top_k=req.top_k,
            model=req.embedding_model,
            sector_filter=req.sector_filter,
            ticker_filter=req.ticker_filter,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("retrieval_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from None

    if not retrieval.chunks:
        raise HTTPException(status_code=404, detail="No matching context found for this query.")

    try:
        generation = generator.generate(req.question, retrieval.chunks)
    except Exception as exc:  # noqa: BLE001
        log.error("generation_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from None

    embedding_model_name = "voyage-finance-2" if req.embedding_model == "voyage" else "text-embedding-3-small"
    embedding_cost = estimate_cost_usd(embedding_model_name, retrieval.embedding_tokens)
    total_cost = embedding_cost + generation.cost_usd
    total_latency_ms = (time.monotonic() - start) * 1000

    log.info(
        "query_served",
        question=req.question,
        embedding_model=req.embedding_model,
        refused=generation.refused,
        cost_usd=round(total_cost, 6),
        latency_ms=round(total_latency_ms, 1),
    )

    return QueryResponse(
        question=req.question,
        answer=generation.answer,
        refused=generation.refused,
        embedding_model=req.embedding_model,
        generation_model=GENERATION_MODEL,
        sources=[
            SourceChunk(
                chunk_id=c.chunk_id,
                score=c.score,
                ticker=c.ticker,
                company_name=c.company_name,
                form=c.form,
                report_date=c.report_date,
                source_url=c.source_url,
                text_preview=c.text[:300],
            )
            for c in retrieval.chunks
        ],
        retrieval_latency_ms=retrieval.latency_ms,
        generation_latency_ms=generation.latency_ms,
        total_latency_ms=total_latency_ms,
        embedding_tokens=retrieval.embedding_tokens,
        generation_input_tokens=generation.input_tokens,
        generation_output_tokens=generation.output_tokens,
        estimated_cost_usd=total_cost,
    )
