"""API tests with retrieval/generation mocked -- no real API calls, no cost."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval_harness.api.main import app  # noqa: E402
from rag_eval_harness.generation.generator import GenerationResult  # noqa: E402
from rag_eval_harness.retrieval.retriever import RetrievalResult, RetrievedChunk  # noqa: E402


@pytest.fixture
def client():
    fake_chunk = RetrievedChunk(
        chunk_id="AAPL_10K_2025-09-27_0100",
        score=0.64,
        text="Apple Inc. total net sales were $416,161 million in fiscal 2025.",
        ticker="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        form="10-K",
        report_date="2025-09-27",
        source_url="https://example.com/aapl10k",
    )
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = RetrievalResult(
        chunks=[fake_chunk], embedding_model="voyage", embedding_tokens=12, latency_ms=42.0
    )
    fake_generator = MagicMock()
    fake_generator.generate.return_value = GenerationResult(
        answer="Apple's total net sales were $416,161 million. (AAPL 10-K, 2025-09-27)",
        model="claude-haiku-4-5",
        input_tokens=500,
        output_tokens=30,
        latency_ms=800.0,
        cost_usd=0.001,
        refused=False,
    )

    with TestClient(app) as test_client:
        app.state.retriever = fake_retriever
        app.state.generator = fake_generator
        app.state.pc = MagicMock()
        app.state.pc.Index.return_value.describe_index_stats.return_value = {
            "total_vector_count": 11872
        }
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_query_returns_grounded_answer(client):
    resp = client.post("/query", json={"question": "What were Apple's net sales?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "416,161" in body["answer"]
    assert body["refused"] is False
    assert len(body["sources"]) == 1
    assert body["sources"][0]["ticker"] == "AAPL"
    assert body["estimated_cost_usd"] > 0


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_404_when_no_chunks_retrieved(client):
    app.state.retriever.retrieve.return_value = RetrievalResult(
        chunks=[], embedding_model="voyage", embedding_tokens=5, latency_ms=10.0
    )
    resp = client.post("/query", json={"question": "anything"})
    assert resp.status_code == 404


def test_metrics_endpoint_exposes_query_counters(client):
    client.post("/query", json={"question": "What were Apple's net sales?"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_eval_harness_query_requests_total" in resp.text
    assert 'rag_eval_harness_query_requests_total{refused="False"}' in resp.text
