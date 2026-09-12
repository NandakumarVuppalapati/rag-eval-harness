"""Regression detection: compare a run's scores against a rolling baseline.

Deliberately pure Python with no I/O, so it's testable without a database
or any live API calls -- the interesting logic here is "is this drop
big enough to page someone", not how the numbers were fetched.
"""

from __future__ import annotations

from dataclasses import dataclass

# All four Ragas metrics and refusal_rate are 0-1, higher-is-better.
MONITORED_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "refusal_rate",
]

# A metric that drops by more than this fraction relative to its recent
# baseline average is flagged. 0.10 = a 10% relative drop.
DEFAULT_DROP_THRESHOLD = 0.10

# Below this many prior runs, a baseline is considered too noisy/thin to
# alert on confidently; regressions are still reported but marked low
# confidence.
MIN_BASELINE_RUNS_FOR_CONFIDENCE = 3


@dataclass(frozen=True)
class RegressionFinding:
    metric: str
    current_value: float
    baseline_mean: float
    relative_drop: float
    is_regression: bool
    confidence: str  # "low" | "normal"


def detect_regressions(
    current: dict,
    baseline_runs: list[dict],
    metrics: list[str] | None = None,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> list[RegressionFinding]:
    """Compare `current` (one run's metric dict) against the mean of
    `baseline_runs` (prior runs' metric dicts, most-recent-first or in
    any order -- all are averaged equally).

    A metric missing from `current` or with no valid baseline values is
    skipped rather than treated as a regression: a metric we can't
    measure is not evidence that it got worse.
    """
    metrics = metrics or MONITORED_METRICS
    confidence = "normal" if len(baseline_runs) >= MIN_BASELINE_RUNS_FOR_CONFIDENCE else "low"

    findings: list[RegressionFinding] = []
    for metric in metrics:
        current_value = current.get(metric)
        if current_value is None:
            continue

        baseline_values = [b[metric] for b in baseline_runs if b.get(metric) is not None]
        if not baseline_values:
            continue
        baseline_mean = sum(baseline_values) / len(baseline_values)
        if baseline_mean <= 0:
            continue

        relative_drop = (baseline_mean - current_value) / baseline_mean
        is_regression = relative_drop > drop_threshold

        findings.append(
            RegressionFinding(
                metric=metric,
                current_value=current_value,
                baseline_mean=baseline_mean,
                relative_drop=relative_drop,
                is_regression=is_regression,
                confidence=confidence,
            )
        )
    return findings


def format_alert(findings: list[RegressionFinding], embedding_model: str) -> str | None:
    """Human-readable alert text for the regressions in `findings`, or
    None if nothing regressed. This is what the nightly Airflow DAG's
    alerting task sends."""
    regressions = [f for f in findings if f.is_regression]
    if not regressions:
        return None

    lines = [f"RAG evaluation regression detected ({embedding_model} embeddings):"]
    for f in regressions:
        confidence_note = " [low confidence: thin baseline]" if f.confidence == "low" else ""
        lines.append(
            f"  - {f.metric}: {f.current_value:.3f} vs baseline {f.baseline_mean:.3f} "
            f"({f.relative_drop * 100:.1f}% drop){confidence_note}"
        )
    return "\n".join(lines)
