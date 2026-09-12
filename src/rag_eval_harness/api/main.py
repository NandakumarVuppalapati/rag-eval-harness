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
from fastapi import FastAPI, HTTPException, Response
from pinecone import Pinecone
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Histogram, generate_latest

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

# --- Prometheus metrics -----------------------------------------------
# Live request-level metrics (always available). Eval-quality gauges are
# added at scrape time in /metrics, read from Postgres when DATABASE_URL is
# set -- there's deliberately no in-process eval state, since the nightly
# Airflow DAG runs in a separate process from this API service.
QUERY_REQUESTS = Counter(
    "rag_eval_harness_query_requests_total",
    "Total /query requests served",
    ["refused"],
)
QUERY_LATENCY_SECONDS = Histogram(
    "rag_eval_harness_query_latency_seconds",
    "End-to-end /query latency in seconds",
)
QUERY_COST_USD = Histogram(
    "rag_eval_harness_query_cost_usd",
    "Estimated cost per /query call in USD",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
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
    QUERY_REQUESTS.labels(refused=str(generation.refused)).inc()
    QUERY_LATENCY_SECONDS.observe(total_latency_ms / 1000)
    QUERY_COST_USD.observe(total_cost)

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


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint. Always exposes live request metrics;
    additionally exposes the most recent nightly-eval scores per
    embedding model when DATABASE_URL points at a reachable Postgres
    instance (set by docker-compose in production). Missing/unreachable
    DB is not an error here -- it just means the eval gauges are absent
    from this scrape, which is the correct behavior for a service that
    can run standalone without the observability stack.
    """
    payload = [generate_latest(REGISTRY)]
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        try:
            from rag_eval_harness.observability.storage import get_engine, get_latest_run

            engine = get_engine(database_url)
            lines = []
            for embedding_model in ("voyage", "openai"):
                run = get_latest_run(engine, embedding_model)
                if not run:
                    continue
                for metric in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "refusal_rate",
                ):
                    value = run.get(metric)
                    if value is not None:
                        lines.append(
                            f'rag_eval_harness_last_eval_{metric}{{embedding_model="{embedding_model}"}} {value}'
                        )
            if lines:
                payload.append(("\n".join(lines) + "\n").encode())
        except Exception as exc:  # noqa: BLE001
            log.warning("metrics_eval_lookup_failed", error=str(exc))
    return Response(content=b"".join(payload), media_type=CONTENT_TYPE_LATEST)
