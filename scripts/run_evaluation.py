"""Run the golden dataset through the RAG pipeline and score it with Ragas.

Runs every answerable golden question (numeric + narrative + cross_document)
through retrieval + generation for one or both embedding models, scores the
answerable slice with Ragas (faithfulness, answer relevancy, context
precision, context recall) using a cross-family OpenAI judge, and scores the
unanswerable slice as a refusal rate. Results are written to
data/eval_results/<timestamp>_<embedding_model>.json.

Usage:
    python scripts/run_evaluation.py --embedding-model voyage
    python scripts/run_evaluation.py --embedding-model both
    python scripts/run_evaluation.py --embedding-model voyage --sample 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import anthropic  # noqa: E402
import openai  # noqa: E402
import voyageai  # noqa: E402
from pinecone import Pinecone  # noqa: E402

from rag_eval_harness.evaluation.harness import (  # noqa: E402
    ANSWERABLE_CATEGORIES,
    load_golden_dataset,
    make_judge,
    results_to_jsonable,
    run_pipeline,
    score_answerable,
    score_refusal,
)
from rag_eval_harness.generation.generator import Generator  # noqa: E402
from rag_eval_harness.retrieval.retriever import Retriever  # noqa: E402

EVAL_RESULTS_DIR = REPO_ROOT / "data" / "eval_results"


def build_clients():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    retriever = Retriever(pc, voyage_client, openai_client)
    generator = Generator(anthropic_client)
    return retriever, generator


def run_for_embedding_model(
    embedding_model: str,
    questions: list[dict],
    retriever,
    generator,
    sample: int | None,
) -> dict:
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

    print(f"Pipeline run complete in {pipeline_elapsed:.1f}s. Scoring with Ragas (cross-family OpenAI judge)...")
    llm, embeddings = make_judge()
    t1 = time.monotonic()
    ragas_scores = score_answerable(answerable_results, llm, embeddings)
    ragas_elapsed = time.monotonic() - t1

    refusal_scores = score_refusal(unanswerable_results)

    total_generation_cost = sum(r.generation_cost_usd for r in all_results)

    return {
        "embedding_model": embedding_model,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_answerable": len(answerable_results),
        "num_unanswerable": len(unanswerable_results),
        "pipeline_elapsed_s": pipeline_elapsed,
        "ragas_elapsed_s": ragas_elapsed,
        "total_generation_cost_usd": total_generation_cost,
        "ragas": ragas_scores,
        "refusal": refusal_scores,
        "raw_results": results_to_jsonable(all_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", choices=["voyage", "openai", "both"], default="voyage")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N answerable questions (smoke-test / cost control)",
    )
    args = parser.parse_args()

    questions = load_golden_dataset()
    retriever, generator = build_clients()

    models = ["voyage", "openai"] if args.embedding_model == "both" else [args.embedding_model]

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for model in models:
        run = run_for_embedding_model(model, questions, retriever, generator, args.sample)
        out_path = EVAL_RESULTS_DIR / f"{run_timestamp}_{model}.json"
        out_path.write_text(json.dumps(run, indent=2))

        agg = run["ragas"]["aggregate"]
        print(f"\n--- {model} summary ---")
        print(f"  Ragas aggregate: {agg}")
        print(f"  Refusal rate on unanswerable set: {run['refusal']['refusal_rate']}")
        print(f"  Total generation cost: ${run['total_generation_cost_usd']:.4f}")
        print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
