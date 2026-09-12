"""Nightly evaluation DAG.

Runs the golden dataset through the RAG pipeline for both embedding
models (voyage-finance-2 and the text-embedding-3-small baseline),
persists the results, and alerts on regressions against each model's
own rolling history.

The two embedding models run as independent branches so a Voyage outage
doesn't block the OpenAI-baseline run (or vice versa); each branch's
"score" task only proceeds to "persist_and_alert" if its own "run"
task succeeded.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL_RESULTS_DIR = REPO_ROOT / "data" / "eval_results"

default_args = {
    "owner": "rag-eval-harness",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _run_evaluation(embedding_model: str, **context) -> str:
    """Run the full golden dataset through retrieval + generation + Ragas
    for one embedding model, write the result JSON, and return its path
    (passed to the next task via XCom -- the run dict itself is too large
    and text-heavy to push through XCom directly)."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    import anthropic
    import openai
    import voyageai
    from pinecone import Pinecone

    import rag_eval_harness  # noqa: F401  applies the ragas/vertexai compat shim
    from rag_eval_harness.evaluation.harness import load_golden_dataset, run_for_embedding_model
    from rag_eval_harness.generation.generator import Generator
    from rag_eval_harness.retrieval.retriever import Retriever

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    retriever = Retriever(pc, voyage_client, openai_client)
    generator = Generator(anthropic_client)

    questions = load_golden_dataset()
    run = run_for_embedding_model(embedding_model, questions, retriever, generator, sample=None)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = context["ts_nodash"]
    out_path = EVAL_RESULTS_DIR / f"{run_timestamp}_{embedding_model}.json"
    out_path.write_text(json.dumps(run, indent=2))
    return str(out_path)


def _persist_and_alert(embedding_model: str, **context) -> None:
    """Load the run this branch just produced into Postgres and check it
    against that embedding model's own recent history."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    from rag_eval_harness.observability.regression import detect_regressions, format_alert
    from rag_eval_harness.observability.storage import get_engine, get_recent_runs, init_db, load_run

    ti = context["ti"]
    run_json_path = ti.xcom_pull(task_ids=f"run_evaluation_{embedding_model}")
    run = json.loads(Path(run_json_path).read_text())

    database_url = os.environ.get("DATABASE_URL", f"sqlite:///{REPO_ROOT}/data/eval_results/eval_results.db")
    engine = get_engine(database_url)
    init_db(engine)

    baseline_runs = get_recent_runs(engine, embedding_model=embedding_model, limit=10)
    load_run(engine, run)

    current_metrics = {**run["ragas"]["aggregate"], "refusal_rate": run["refusal"]["refusal_rate"]}
    findings = detect_regressions(current_metrics, baseline_runs)
    alert = format_alert(findings, embedding_model)
    if alert:
        # A real on-call setup would page/Slack/email here instead of just
        # raising -- raising is enough to make Airflow mark the task
        # (and therefore the run) as failed and surface it in the UI.
        raise RuntimeError(alert)


with DAG(
    dag_id="rag_eval_harness_nightly",
    description="Nightly golden-dataset evaluation for both embedding models, with regression alerting.",
    default_args=default_args,
    schedule="0 6 * * *",  # 06:00 UTC nightly
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["rag-eval-harness", "evaluation"],
) as dag:
    for embedding_model in ("voyage", "openai"):
        run_task = PythonOperator(
            task_id=f"run_evaluation_{embedding_model}",
            python_callable=_run_evaluation,
            op_kwargs={"embedding_model": embedding_model},
        )
        persist_task = PythonOperator(
            task_id=f"persist_and_alert_{embedding_model}",
            python_callable=_persist_and_alert,
            op_kwargs={"embedding_model": embedding_model},
        )
        run_task >> persist_task
