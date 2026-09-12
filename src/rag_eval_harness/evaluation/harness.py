"""Ragas-based evaluation harness.

Runs the golden dataset (data/golden_dataset/) through the retrieval +
generation pipeline and scores it with Ragas. Judging is deliberately
cross-family: generation happens on Claude Haiku (see
rag_eval_harness.generation.generator), but the Ragas judge LLM here is
OpenAI's gpt-4o-mini. If the same model family both answered and graded
the question, a systematic failure mode shared by that family (a
plausible-sounding but wrong number, say) would be invisible to the
judge too. A different family is far less likely to share that blind
spot.

The unanswerable slice of the golden dataset is scored separately, as a
refusal rate, rather than through Ragas: Ragas' metrics assume there is
a correct answer to compare against, which is exactly what these
questions don't have.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GOLDEN_DIR = REPO_ROOT / "data" / "golden_dataset"

JUDGE_LLM_MODEL = "gpt-4o-mini"
JUDGE_EMBED_MODEL = "text-embedding-3-small"

EmbeddingModelName = Literal["voyage", "openai"]


@dataclass
class QuestionResult:
    """One golden question run through the pipeline once."""

    id: str
    category: str
    question: str
    reference_answer: str
    embedding_model: str
    answer: str
    refused: bool
    contexts: list[str]
    retrieved_tickers: list[str]
    retrieval_latency_ms: float
    generation_latency_ms: float
    generation_input_tokens: int
    generation_output_tokens: int
    generation_cost_usd: float

    def to_ragas_sample(self) -> dict:
        return {
            "user_input": self.question,
            "response": self.answer,
            "retrieved_contexts": self.contexts,
            "reference": self.reference_answer,
        }


def load_golden_dataset() -> list[dict]:
    """Load and tag every golden question with its source file's category."""
    numeric = json.loads((GOLDEN_DIR / "numeric_questions.json").read_text())
    narrative = json.loads((GOLDEN_DIR / "narrative_questions.json").read_text())
    unanswerable = json.loads((GOLDEN_DIR / "unanswerable_questions.json").read_text())
    return numeric + narrative + unanswerable


ANSWERABLE_CATEGORIES = {"numeric_xbrl", "narrative", "cross_document"}


def run_pipeline(
    question: dict,
    retriever,
    generator,
    embedding_model: EmbeddingModelName,
    top_k: int = 5,
) -> QuestionResult:
    """Run one golden question through retrieval + generation exactly once."""
    retrieval = retriever.retrieve(question["question"], top_k=top_k, model=embedding_model)
    generation = generator.generate(question["question"], retrieval.chunks)
    return QuestionResult(
        id=question["id"],
        category=question["category"],
        question=question["question"],
        reference_answer=question.get("reference_answer", ""),
        embedding_model=embedding_model,
        answer=generation.answer,
        refused=generation.refused,
        contexts=[c.text for c in retrieval.chunks],
        retrieved_tickers=[c.ticker for c in retrieval.chunks],
        retrieval_latency_ms=retrieval.latency_ms,
        generation_latency_ms=generation.latency_ms,
        generation_input_tokens=generation.input_tokens,
        generation_output_tokens=generation.output_tokens,
        generation_cost_usd=generation.cost_usd,
    )


def make_judge():
    """Build the cross-family Ragas judge (OpenAI), independent of the
    Claude-Haiku generator being evaluated."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_LLM_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBED_MODEL))
    return llm, embeddings


def score_answerable(results: list[QuestionResult], llm, embeddings) -> dict:
    """Score the answerable slice (numeric + narrative + cross_document)
    with Ragas' faithfulness / answer_relevancy / context_precision /
    context_recall metrics. Returns per-question scores plus the
    dataset-level aggregate Ragas computes."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    if not results:
        return {"per_question": [], "aggregate": {}}

    dataset = EvaluationDataset.from_list([r.to_ragas_sample() for r in results])
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
    )
    scores_df = ragas_result.to_pandas()
    per_question = []
    for i, r in enumerate(results):
        row = scores_df.iloc[i]
        per_question.append(
            {
                "id": r.id,
                "category": r.category,
                "embedding_model": r.embedding_model,
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
                "context_recall": _safe_float(row.get("context_recall")),
            }
        )
    aggregate = {k: _safe_float(v) for k, v in dict(ragas_result).items()}
    return {"per_question": per_question, "aggregate": aggregate}


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None


def score_refusal(results: list[QuestionResult]) -> dict:
    """Refusal accuracy for the unanswerable slice: did the generator
    actually decline, rather than fabricate a confident-sounding answer?"""
    if not results:
        return {"count": 0, "refusal_rate": None, "failures": []}
    refused_count = sum(1 for r in results if r.refused)
    failures = [
        {"id": r.id, "question": r.question, "answer": r.answer}
        for r in results
        if not r.refused
    ]
    return {
        "count": len(results),
        "refusal_rate": refused_count / len(results),
        "failures": failures,
    }


def results_to_jsonable(results: list[QuestionResult]) -> list[dict]:
    return [asdict(r) for r in results]
