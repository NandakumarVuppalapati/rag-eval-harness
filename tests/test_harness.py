"""Tests for score_answerable's aggregate computation.

Regression coverage for a real bug: ragas 0.4.3's EvaluationResult only
implements __getitem__(key: str) and has no keys()/__iter__, so
dict(ragas_result) falls back to integer-indexed iteration and raises
KeyError: 0 immediately. This was never caught by testing because this
code path had never actually run to completion against real data before
(every prior attempt hit the OpenAI rate limit before reaching it) -- see
harness.py's score_answerable for the fix and full explanation.
"""

from __future__ import annotations

import pandas as pd

from rag_eval_harness.evaluation.harness import QuestionResult, score_answerable


def _question_result(id_: str, embedding_model: str = "voyage") -> QuestionResult:
    return QuestionResult(
        id=id_,
        category="numeric_xbrl",
        question=f"Question for {id_}",
        reference_answer="42",
        embedding_model=embedding_model,
        answer="42",
        refused=False,
        contexts=["some context"],
        retrieved_tickers=["AAPL"],
        retrieval_latency_ms=1.0,
        generation_latency_ms=1.0,
        generation_input_tokens=10,
        generation_output_tokens=5,
        generation_cost_usd=0.001,
    )


class _FakeEvaluationResult:
    """Mirrors the real ragas 0.4.3 EvaluationResult closely enough to
    reproduce the bug: __getitem__ only accepts string metric names, so
    dict(result) (integer-indexed iteration) raises KeyError."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(key)
        return self._df[key].tolist()


def test_score_answerable_computes_aggregate_from_per_question_scores(monkeypatch):
    fake_df = pd.DataFrame(
        {
            "faithfulness": [1.0, 0.5],
            "answer_relevancy": [0.9, 0.7],
            "context_precision": [0.8, 0.6],
            "context_recall": [0.85, 0.65],
        }
    )
    monkeypatch.setattr("ragas.evaluate", lambda **kwargs: _FakeEvaluationResult(fake_df))

    results = [_question_result("numeric_000"), _question_result("numeric_001")]
    scores = score_answerable(results, llm=object(), embeddings=object())

    assert scores["aggregate"] == {
        "faithfulness": 0.75,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
        "context_recall": 0.75,
    }
    assert len(scores["per_question"]) == 2
    assert scores["per_question"][0]["id"] == "numeric_000"
    assert scores["per_question"][0]["faithfulness"] == 1.0
    assert scores["per_question"][1]["faithfulness"] == 0.5


def test_score_answerable_returns_empty_for_no_results():
    scores = score_answerable([], llm=object(), embeddings=object())
    assert scores == {"per_question": [], "aggregate": {}}
