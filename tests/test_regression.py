from rag_eval_harness.observability.regression import (
    detect_regressions,
    format_alert,
)


def test_no_regression_when_scores_stable():
    current = {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82, "refusal_rate": 1.0}
    baseline = [
        {"faithfulness": 0.91, "answer_relevancy": 0.84, "context_precision": 0.79, "context_recall": 0.81, "refusal_rate": 1.0},
        {"faithfulness": 0.89, "answer_relevancy": 0.86, "context_precision": 0.81, "context_recall": 0.83, "refusal_rate": 0.9},
        {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82, "refusal_rate": 1.0},
    ]
    findings = detect_regressions(current, baseline)
    assert all(not f.is_regression for f in findings)
    assert format_alert(findings, "voyage") is None


def test_clear_regression_is_flagged():
    current = {"faithfulness": 0.50, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82, "refusal_rate": 1.0}
    baseline = [
        {"faithfulness": 0.90, "answer_relevancy": 0.85, "context_precision": 0.80, "context_recall": 0.82, "refusal_rate": 1.0}
        for _ in range(3)
    ]
    findings = detect_regressions(current, baseline)
    faithfulness_finding = next(f for f in findings if f.metric == "faithfulness")
    assert faithfulness_finding.is_regression
    assert faithfulness_finding.confidence == "normal"

    alert = format_alert(findings, "voyage")
    assert alert is not None
    assert "faithfulness" in alert


def test_small_fluctuation_under_threshold_is_not_a_regression():
    current = {"faithfulness": 0.87}
    baseline = [{"faithfulness": 0.90}, {"faithfulness": 0.89}, {"faithfulness": 0.91}]
    findings = detect_regressions(current, baseline)
    assert len(findings) == 1
    assert not findings[0].is_regression


def test_cold_start_with_no_baseline_reports_nothing():
    current = {"faithfulness": 0.10}
    findings = detect_regressions(current, baseline_runs=[])
    assert findings == []
    assert format_alert(findings, "voyage") is None


def test_missing_metric_in_current_is_skipped_not_flagged():
    current = {"faithfulness": 0.9}
    baseline = [{"faithfulness": 0.9, "answer_relevancy": 0.8}]
    findings = detect_regressions(current, baseline)
    assert {f.metric for f in findings} == {"faithfulness"}


def test_thin_baseline_marks_low_confidence():
    current = {"faithfulness": 0.5}
    baseline = [{"faithfulness": 0.9}]  # only one prior run
    findings = detect_regressions(current, baseline)
    assert findings[0].is_regression
    assert findings[0].confidence == "low"
