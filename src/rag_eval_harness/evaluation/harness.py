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
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# REPO_ROOT (four parents up from this file) only lands on the real repo
# root for a source checkout / editable install (`pip install -e .`), where
# __file__ is still src/rag_eval_harness/evaluation/harness.py -- that's
# true for local dev and CI. A real `pip install <package>` (no -e), which
# is what both docker/Dockerfile and docker/airflow.Dockerfile do, copies
# this file into site-packages instead, where four-parents-up lands
# somewhere under the venv, not the repo -- discovered by watching a real
# Airflow task fail with FileNotFoundError for
# .../site-packages/../../data/golden_dataset/numeric_questions.json once
# the DAG's imports were finally fixed (see run_for_embedding_model's
# docstring below for that saga). data/golden_dataset/ *is* present inside
# the Airflow containers -- docker-compose.yml bind-mounts the whole repo
# data/ dir to /opt/airflow/data -- just not where this __file__-relative
# guess looks for it, hence GOLDEN_DATASET_DIR as an explicit override
# docker-compose.yml's airflow-common environment block sets.
GOLDEN_DIR = (
    Path(os.environ["GOLDEN_DATASET_DIR"])
    if "GOLDEN_DATASET_DIR" in os.environ
    else REPO_ROOT / "data" / "golden_dataset"
)

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


_RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def score_answerable(results: list[QuestionResult], llm, embeddings) -> dict:
    """Score the answerable slice (numeric + narrative + cross_document)
    with Ragas' faithfulness / answer_relevancy / context_precision /
    context_recall metrics. Returns per-question scores plus the
    dataset-level aggregate (the mean of each metric's per-question
    scores, computed here rather than trusting Ragas's own aggregate
    object -- see the note below)."""
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
                **{name: _safe_float(row.get(name)) for name in _RAGAS_METRIC_NAMES},
            }
        )
    # NOTE: an earlier version of this function called dict(ragas_result) to
    # get the dataset-level aggregate. Ragas 0.4.3's EvaluationResult only
    # implements __getitem__(key: str) -- no keys()/__iter__ -- so dict()
    # falls back to integer-indexed iteration and raises KeyError: 0 on the
    # very first call. That path was never actually exercised end-to-end
    # before real data made it this far (every prior run hit the OpenAI
    # rate limit first), so the bug shipped invisibly. Computing the mean
    # ourselves from scores_df avoids depending on that unstable API at all.
    aggregate = {
        name: _safe_float(scores_df[name].mean()) if name in scores_df.columns else None
        for name in _RAGAS_METRIC_NAMES
    }
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


def question_results_from_jsonable(raw_results: list[dict]) -> list[QuestionResult]:
    """Inverse of results_to_jsonable. Used by scripts/score_ragas.py to
    re-score a run that was persisted with --skip-ragas (see that flag's
    docstring in scripts/run_evaluation.py for why that split exists)
    without re-running retrieval or generation.
    """
    return [QuestionResult(**d) for d in raw_results]


def run_for_embedding_model(
    embedding_model: EmbeddingModelName,
    questions: list[dict],
    retriever,
    generator,
    sample: int | None,
    score_with_ragas: bool = True,
) -> dict:
    """Run every golden question through the pipeline once for one
    embedding model, optionally score it with Ragas, and assemble the
    aggregate run dict that scripts/run_evaluation.py writes to
    data/eval_results/<timestamp>_<embedding_model>.json and
    dags/nightly_evaluation_dag.py's _run_evaluation task produces for its
    XCom-passed path. Lives here (not in scripts/run_evaluation.py, where
    it was originally written) because the Airflow image only pip installs
    this package -- see docker/airflow.Dockerfile -- it never gets a copy
    of scripts/, and the DAG's own REPO_ROOT computation
    (Path(__file__).resolve().parent.parent, which is /opt/airflow inside
    that image, not the repo root) can't be pointed at it either way.
    scripts/run_evaluation.py now imports this instead of defining its own
    copy, so its CLI keeps working unchanged.
    """
    answerable_qs = [q for q in questions if q["category"] in ANSWERABLE_CATEGORIES]
    unanswerable_qs = [q for q in questions if q["category"] == "unanswerable"]
    if sample:
        answerable_qs = answerable_qs[:sample]
        unanswerable_qs = unanswerable_qs[: max(1, sample // 4)]

    print(f"\n=== embedding_model={embedding_model}: running {len(answerable_qs)} answerable + "
          f"{len(unanswerable_qs)} unanswerable questions through the pipeline ===")

    all_results = []
    t0 = time.monotonic()
    for i, q in enumerate(answerable_qs + unanswerable_qs):
        r = run_pipeline(q, retriever, generator, embedding_model=embedding_model)
        all_results.append(r)
        kind = "answerable" if q["category"] in ANSWERABLE_CATEGORIES else "unanswerable"
        print(f"  [{i + 1}/{len(answerable_qs) + len(unanswerable_qs)}] {r.id} ({kind}, {r.category}) "
              f"refused={r.refused} retrieval={r.retrieval_latency_ms:.0f}ms gen={r.generation_latency_ms:.0f}ms")
    pipeline_elapsed = time.monotonic() - t0

    answerable_results = [r for r in all_results if r.category in ANSWERABLE_CATEGORIES]
    unanswerable_results = [r for r in all_results if r.category == "unanswerable"]

    if score_with_ragas:
        print(f"Pipeline run complete in {pipeline_elapsed:.1f}s. Scoring with Ragas (cross-family OpenAI judge)...")
        llm, embeddings = make_judge()
        t1 = time.monotonic()
        ragas_scores = score_answerable(answerable_results, llm, embeddings)
        ragas_elapsed = time.monotonic() - t1
    else:
        print(f"Pipeline run complete in {pipeline_elapsed:.1f}s. Skipping Ragas scoring (--skip-ragas); "
              "run scripts/score_ragas.py on the output file to backfill it later.")
        ragas_scores = {"per_question": [], "aggregate": {}}
        ragas_elapsed = 0.0

    refusal_scores = score_refusal(unanswerable_results)

    total_generation_cost = sum(r.generation_cost_usd for r in all_results)

    return {
        "embedding_model": embedding_model,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_answerable": len(answerable_results),
        "num_unanswerable": len(unanswerable_results),
        "pipeline_elapsed_s": pipeline_elapsed,
        "ragas_elapsed_s": ragas_elapsed,
        "ragas_pending": not score_with_ragas,
        "total_generation_cost_usd": total_generation_cost,
        "ragas": ragas_scores,
        "refusal": refusal_scores,
        "raw_results": results_to_jsonable(all_results),
    }
