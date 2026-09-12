from datetime import datetime, timezone

from rag_eval_harness.observability.storage import (
    get_engine,
    get_latest_run,
    get_recent_runs,
    init_db,
    load_run,
)


def _sample_run(embedding_model: str = "voyage", faithfulness: float = 0.9) -> dict:
    return {
        "embedding_model": embedding_model,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_answerable": 2,
        "num_unanswerable": 1,
        "pipeline_elapsed_s": 12.3,
        "ragas_elapsed_s": 45.6,
        "total_generation_cost_usd": 0.01,
        "ragas": {
            "aggregate": {
                "faithfulness": faithfulness,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.75,
            },
            "per_question": [
                {
                    "id": "numeric_000",
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "context_precision": 0.8,
                    "context_recall": 0.85,
                },
                {
                    "id": "numeric_001",
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.7,
                    "context_precision": 0.6,
                    "context_recall": 0.65,
                },
            ],
        },
        "refusal": {"refusal_rate": 1.0, "count": 1, "failures": []},
        "raw_results": [
            {
                "id": "numeric_000",
                "category": "numeric_xbrl",
                "embedding_model": embedding_model,
                "refused": False,
                "retrieval_latency_ms": 300.0,
                "generation_latency_ms": 1500.0,
                "generation_cost_usd": 0.002,
                "answer": "The answer is $1.",
            },
            {
                "id": "numeric_001",
                "category": "numeric_xbrl",
                "embedding_model": embedding_model,
                "refused": False,
                "retrieval_latency_ms": 280.0,
                "generation_latency_ms": 1600.0,
                "generation_cost_usd": 0.002,
                "answer": "The answer is $2.",
            },
            {
                "id": "unanswerable_000",
                "category": "unanswerable",
                "embedding_model": embedding_model,
                "refused": True,
                "retrieval_latency_ms": 250.0,
                "generation_latency_ms": 900.0,
                "generation_cost_usd": 0.001,
                "answer": "I cannot answer this based on the provided context.",
            },
        ],
    }


def test_load_run_and_read_back():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)

    run_id = load_run(engine, _sample_run())
    assert run_id == 1

    latest = get_latest_run(engine, "voyage")
    assert latest is not None
    assert latest["embedding_model"] == "voyage"
    assert latest["faithfulness"] == 0.9
    assert latest["num_answerable"] == 2
    assert latest["num_unanswerable"] == 1


def test_get_recent_runs_orders_newest_first_and_filters_by_model():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)

    load_run(engine, _sample_run(embedding_model="voyage", faithfulness=0.5))
    load_run(engine, _sample_run(embedding_model="openai", faithfulness=0.6))
    load_run(engine, _sample_run(embedding_model="voyage", faithfulness=0.95))

    voyage_runs = get_recent_runs(engine, embedding_model="voyage", limit=10)
    assert len(voyage_runs) == 2
    assert all(r["embedding_model"] == "voyage" for r in voyage_runs)

    all_runs = get_recent_runs(engine, limit=10)
    assert len(all_runs) == 3


def test_get_latest_run_returns_none_when_no_runs_for_model():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    load_run(engine, _sample_run(embedding_model="voyage"))

    assert get_latest_run(engine, "openai") is None
