"""Persistence layer for evaluation runs.

Uses SQLAlchemy Core with a deliberately backend-agnostic schema (no
Postgres-specific types) so the exact same code path can be exercised
against an in-memory SQLite engine in tests, while production points
DATABASE_URL at the Postgres instance docker-compose brings up. See
docs/adr/0003-observability-storage.md for why.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

eval_runs = Table(
    "eval_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("embedding_model", String(32), nullable=False),
    Column("run_at_utc", DateTime, nullable=False),
    Column("num_answerable", Integer, nullable=False),
    Column("num_unanswerable", Integer, nullable=False),
    Column("faithfulness", Float, nullable=True),
    Column("answer_relevancy", Float, nullable=True),
    Column("context_precision", Float, nullable=True),
    Column("context_recall", Float, nullable=True),
    Column("refusal_rate", Float, nullable=True),
    Column("total_generation_cost_usd", Float, nullable=False),
    Column("pipeline_elapsed_s", Float, nullable=False),
    Column("ragas_elapsed_s", Float, nullable=False),
)

eval_question_results = Table(
    "eval_question_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, nullable=False),
    Column("question_id", String(64), nullable=False),
    Column("category", String(32), nullable=False),
    Column("embedding_model", String(32), nullable=False),
    Column("refused", Boolean, nullable=False),
    Column("faithfulness", Float, nullable=True),
    Column("answer_relevancy", Float, nullable=True),
    Column("context_precision", Float, nullable=True),
    Column("context_recall", Float, nullable=True),
    Column("retrieval_latency_ms", Float, nullable=False),
    Column("generation_latency_ms", Float, nullable=False),
    Column("generation_cost_usd", Float, nullable=False),
    Column("answer", Text, nullable=False),
)


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def load_run(engine: Engine, run: dict) -> int:
    """Persist one run_evaluation.py output dict (see
    rag_eval_harness.evaluation.harness.run_for_embedding_model's return
    value) as one eval_runs row plus one eval_question_results row per
    question. Returns the new run's id."""
    agg = run.get("ragas", {}).get("aggregate", {}) or {}
    per_question = {q["id"]: q for q in run.get("ragas", {}).get("per_question", [])}
    refusal = run.get("refusal", {}) or {}

    run_at = run["run_at_utc"]
    if isinstance(run_at, str):
        run_at = datetime.fromisoformat(run_at)

    with engine.begin() as conn:
        result = conn.execute(
            insert(eval_runs).values(
                embedding_model=run["embedding_model"],
                run_at_utc=run_at,
                num_answerable=run["num_answerable"],
                num_unanswerable=run["num_unanswerable"],
                faithfulness=agg.get("faithfulness"),
                answer_relevancy=agg.get("answer_relevancy"),
                context_precision=agg.get("context_precision"),
                context_recall=agg.get("context_recall"),
                refusal_rate=refusal.get("refusal_rate"),
                total_generation_cost_usd=run["total_generation_cost_usd"],
                pipeline_elapsed_s=run["pipeline_elapsed_s"],
                ragas_elapsed_s=run["ragas_elapsed_s"],
            )
        )
        run_id = result.inserted_primary_key[0]

        rows = []
        for raw in run.get("raw_results", []):
            scores = per_question.get(raw["id"], {})
            rows.append(
                {
                    "run_id": run_id,
                    "question_id": raw["id"],
                    "category": raw["category"],
                    "embedding_model": raw["embedding_model"],
                    "refused": raw["refused"],
                    "faithfulness": scores.get("faithfulness"),
                    "answer_relevancy": scores.get("answer_relevancy"),
                    "context_precision": scores.get("context_precision"),
                    "context_recall": scores.get("context_recall"),
                    "retrieval_latency_ms": raw["retrieval_latency_ms"],
                    "generation_latency_ms": raw["generation_latency_ms"],
                    "generation_cost_usd": raw["generation_cost_usd"],
                    "answer": raw["answer"],
                }
            )
        if rows:
            conn.execute(insert(eval_question_results), rows)

    return run_id


def get_recent_runs(engine: Engine, embedding_model: str | None = None, limit: int = 10) -> list[dict]:
    """Most recent runs first, optionally filtered to one embedding model."""
    stmt = select(eval_runs).order_by(eval_runs.c.run_at_utc.desc()).limit(limit)
    if embedding_model:
        stmt = stmt.where(eval_runs.c.embedding_model == embedding_model)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def get_latest_run(engine: Engine, embedding_model: str) -> dict | None:
    runs = get_recent_runs(engine, embedding_model=embedding_model, limit=1)
    return runs[0] if runs else None
