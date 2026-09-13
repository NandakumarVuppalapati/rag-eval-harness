"""Run the golden dataset through the RAG pipeline and score it with Ragas.

Runs every answerable golden question (numeric + narrative + cross_document)
through retrieval + generation for one or both embedding models, scores the
answerable slice with Ragas (faithfulness, answer relevancy, context
precision, context recall) using a cross-family OpenAI judge, and scores the
unanswerable slice as a refusal rate. Results are written to
data/eval_results/<timestamp>_<embedding_model>.json.

Retrieval + generation (Pinecone/Voyage/Claude) and Ragas scoring (the
OpenAI judge) are deliberately separable via --skip-ragas: they hit
different providers with different rate-limit behavior, and the pipeline
run is the expensive, non-idempotent half (real Claude generations). A
transient judge-provider outage shouldn't cost a re-run of the pipeline
just to get scored -- run with --skip-ragas, then backfill scores later
with scripts/score_ragas.py once the judge is reachable again. This is
also how dags/nightly_evaluation_dag.py splits the two into separate
Airflow tasks, so a retry only re-does the half that actually failed.

Usage:
    python scripts/run_evaluation.py --embedding-model voyage
    python scripts/run_evaluation.py --embedding-model both
    python scripts/run_evaluation.py --embedding-model voyage --sample 5
    python scripts/run_evaluation.py --embedding-model voyage --skip-ragas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    load_golden_dataset,
    run_for_embedding_model,
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


# run_for_embedding_model now lives in rag_eval_harness.evaluation.harness
# (imported above) -- moved there so dags/nightly_evaluation_dag.py can use
# the exact same function without needing scripts/ inside the Airflow
# image, which only pip installs this package (see docker/airflow.Dockerfile
# and the docstring on harness.run_for_embedding_model for the full story).

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", choices=["voyage", "openai", "both"], default="voyage")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N answerable questions (smoke-test / cost control)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Run retrieval + generation but skip Ragas scoring (see module docstring)",
    )
    args = parser.parse_args()

    questions = load_golden_dataset()
    retriever, generator = build_clients()

    models = ["voyage", "openai"] if args.embedding_model == "both" else [args.embedding_model]

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for model in models:
        run = run_for_embedding_model(
            model, questions, retriever, generator, args.sample, score_with_ragas=not args.skip_ragas
        )
        out_path = EVAL_RESULTS_DIR / f"{run_timestamp}_{model}.json"
        out_path.write_text(json.dumps(run, indent=2))

        agg = run["ragas"]["aggregate"]
        print(f"\n--- {model} summary ---")
        if run["ragas_pending"]:
            print("  Ragas scoring skipped (--skip-ragas) -- run scripts/score_ragas.py on this file later")
        else:
            print(f"  Ragas aggregate: {agg}")
        print(f"  Refusal rate on unanswerable set: {run['refusal']['refusal_rate']}")
        print(f"  Total generation cost: ${run['total_generation_cost_usd']:.4f}")
        print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
