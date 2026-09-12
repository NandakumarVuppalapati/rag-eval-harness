"""Load a run_evaluation.py output JSON file into Postgres and check for
regressions against its recent history.

This is the second half of the nightly job (see dags/nightly_evaluation_dag.py):
run_evaluation.py produces the JSON, this script persists it and decides
whether to alert.

Usage:
    python scripts/load_eval_results.py data/eval_results/20260101T000000Z_voyage.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from rag_eval_harness.observability.regression import detect_regressions, format_alert  # noqa: E402
from rag_eval_harness.observability.storage import (  # noqa: E402
    get_engine,
    get_recent_runs,
    init_db,
    load_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json_path", type=Path)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "sqlite:///./data/eval_results/eval_results.db"),
        help="Defaults to $DATABASE_URL, falling back to a local SQLite file for dev use "
        "without docker-compose.",
    )
    args = parser.parse_args()

    run = json.loads(args.run_json_path.read_text())

    engine = get_engine(args.database_url)
    init_db(engine)

    # Baseline = prior runs for this embedding model, excluding the one we're
    # about to insert.
    baseline_runs = get_recent_runs(engine, embedding_model=run["embedding_model"], limit=10)

    run_id = load_run(engine, run)
    print(f"Persisted run {run_id} ({run['embedding_model']}) to {args.database_url}")

    current_metrics = {
        **run["ragas"]["aggregate"],
        "refusal_rate": run["refusal"]["refusal_rate"],
    }
    findings = detect_regressions(current_metrics, baseline_runs)
    alert = format_alert(findings, run["embedding_model"])

    if alert:
        print("\n" + alert)
        # A real deployment would page/Slack/email here. Printing to stdout
        # is what the Airflow task log captures, which is enough for a
        # nightly batch job with no on-call rotation behind it yet.
    else:
        confidence_note = "" if len(baseline_runs) >= 3 else " (thin baseline, low confidence)"
        print(f"\nNo regressions detected against {len(baseline_runs)} prior run(s){confidence_note}.")


if __name__ == "__main__":
    main()
