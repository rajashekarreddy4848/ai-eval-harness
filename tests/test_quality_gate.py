"""Unit tests for the quality gate (no API calls)."""

import json

from evals.quality_gate import evaluate, markdown, pass_rate, ragas_scores

THRESHOLDS = {"faithfulness": 0.8, "golden_accuracy": 0.85}


def test_scores_at_or_above_threshold_pass():
    rows = evaluate({"faithfulness": 0.8, "golden_accuracy": 0.9}, THRESHOLDS)
    assert all(r["passed"] for r in rows)


def test_score_below_threshold_fails():
    rows = evaluate({"faithfulness": 0.79, "golden_accuracy": 0.9}, THRESHOLDS)
    assert [r["passed"] for r in rows] == [False, True]


def test_missing_score_fails_instead_of_passing_silently():
    rows = evaluate({"faithfulness": 0.9}, THRESHOLDS)
    assert rows[1]["score"] is None and not rows[1]["passed"]


def test_markdown_table_marks_failures():
    table = markdown(evaluate({"faithfulness": 0.5}, {"faithfulness": 0.8}))
    assert "❌ fail" in table and "0.50" in table


def test_ragas_scores_average_and_skip_blank(tmp_path):
    report = tmp_path / "r.csv"
    report.write_text("id,faithfulness,context_precision,context_recall\na,1.0,0.5,1\nb,0.5,,1\n", encoding="utf-8")
    scores = ragas_scores(report)
    assert scores == {"faithfulness": 0.75, "context_precision": 0.5, "context_recall": 1.0}


def test_pass_rate_ignores_known_issue_cases(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "A"}, {"id": "B", "known_issue": "retrieval"}]), encoding="utf-8")
    results = tmp_path / "res.json"
    results.write_text(json.dumps([
        {"id": "A", "passes": 3, "runs": [{}, {}, {}]},
        {"id": "B", "passes": 0, "runs": [{}, {}, {}]},
    ]), encoding="utf-8")
    assert pass_rate(results, cases) == 1.0
