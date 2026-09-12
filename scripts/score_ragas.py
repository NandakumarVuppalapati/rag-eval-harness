"""Backfill Ragas scores into a run JSON produced with run_evaluation.py --skip-ragas.

Retrieval and generation are the expensive, non-idempotent half of an
evaluation run (real Claude Haiku calls against a real corpus); Ragas
scoring is a second, independent API dependency (the OpenAI judge) that can
be transiently rate-limited or down without that pipeline run needing to be
redone. This script re-scores an existing run's raw_results in place,
without touching retrieval or generation at all.

Usage:
    python scripts/score_ragas.py data/eval_results/20260101T000000Z_voyage.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from rag_eval_harness.evaluation.harness import (  # noqa: E402
    ANSWERABLE_CATEGORIES,
    make_judge,
    question_results_from_jsonable,
    score_answerable,
    score_refusal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json_path", type=Path)
    args = parser.parse_args()

    run = json.loads(args.run_json_path.read_text())
    all_results = question_results_from_jsonable(run["raw_results"])
    answerable_results = [r for r in all_results if r.category in ANSWERABLE_CATEGORIES]
    unanswerable_results = [r for r in all_results if r.category == "unanswerable"]

    print(f"Scoring {len(answerable_results)} answerable questions from {args.run_json_path} with Ragas...")
    llm, embeddings = make_judge()
    t0 = time.monotonic()
    ragas_scores = score_answerable(answerable_results, llm, embeddings)
    ragas_elapsed = time.monotonic() - t0

    run["ragas"] = ragas_scores
    run["ragas_elapsed_s"] = ragas_elapsed
    run["ragas_pending"] = False
    # Refusal scoring never needed the judge, but recomputing it here is
    # free (pure Python over already-fetched answers) and keeps this script
    # a complete "finish scoring this run" step rather than a partial one.
    run["refusal"] = score_refusal(unanswerable_results)

    args.run_json_path.write_text(json.dumps(run, indent=2))

    print(f"Ragas aggregate: {ragas_scores['aggregate']}")
    print(f"Refusal rate on unanswerable set: {run['refusal']['refusal_rate']}")
    print(f"Updated {args.run_json_path} in place ({ragas_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
