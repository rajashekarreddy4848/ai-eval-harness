"""Quality gate: compare eval scores to thresholds, print a table, fail below them.

Reads the outputs of the eval suites (no API calls):
  ragas_report.csv                       faithfulness, context precision/recall (tests/test_ragas.py)
  evals/results/pass_rate_golden_*.json  golden-set accuracy (evals/pass_rate.py --set golden)
  evals/results/pass_rate_redteam_*.json red-team block rate (evals/pass_rate.py)
Cases marked "known_issue" are left out of the pass rates, as in the CI gating runs.

Run with: python evals/quality_gate.py [--markdown scores.md]
Exit code 1 if any score is below its threshold in evals/thresholds.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"


def ragas_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = {}
    for metric in ("faithfulness", "context_precision", "context_recall"):
        values = [float(r[metric]) for r in rows if r.get(metric) not in (None, "", "nan")]
        if values:
            scores[metric] = sum(values) / len(values)
    return scores


def pass_rate(results_path: Path, cases_path: Path) -> float | None:
    """Share of runs that passed, over cases without a known_issue."""
    known = {c["id"] for c in json.loads(cases_path.read_text(encoding="utf-8")) if c.get("known_issue")}
    rows = [r for r in json.loads(results_path.read_text(encoding="utf-8")) if r["id"] not in known]
    runs = sum(len(r["runs"]) for r in rows)
    return sum(r["passes"] for r in rows) / runs if runs else None


def latest(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def collect() -> dict[str, float]:
    scores = {}
    report = ROOT / "ragas_report.csv"
    if report.exists():
        scores.update(ragas_scores(report))
    golden = latest("pass_rate_golden_*.json")
    if golden:
        scores["golden_accuracy"] = pass_rate(golden, ROOT / "evals" / "golden_set.json")
    redteam = latest("pass_rate_redteam_*.json")
    if redteam:
        scores["redteam_block_rate"] = pass_rate(redteam, ROOT / "evals" / "redteam_set.json")
    return {k: v for k, v in scores.items() if v is not None}


def evaluate(scores: dict[str, float], thresholds: dict[str, float]) -> list[dict]:
    """One row per threshold. A score that was not produced counts as a failure."""
    rows = []
    for metric, minimum in thresholds.items():
        score = scores.get(metric)
        rows.append({"metric": metric, "score": score, "threshold": minimum,
                     "passed": score is not None and score >= minimum})
    return rows


def markdown(rows: list[dict]) -> str:
    lines = ["| Metric | Score | Threshold | Result |", "|---|---|---|---|"]
    for r in rows:
        score = "missing" if r["score"] is None else f"{r['score']:.2f}"
        lines.append(f"| {r['metric']} | {score} | {r['threshold']:.2f} | {'✅ pass' if r['passed'] else '❌ fail'} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default=str(ROOT / "evals" / "thresholds.json"))
    parser.add_argument("--markdown", metavar="FILE", help="also write the score table to this file")
    args = parser.parse_args()

    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    rows = evaluate(collect(), thresholds)
    table = markdown(rows)
    print(table)
    if args.markdown:
        Path(args.markdown).write_text("### AI quality gate\n\n" + table + "\n", encoding="utf-8")
    if not all(r["passed"] for r in rows):
        print("\nQuality gate FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nQuality gate passed")


if __name__ == "__main__":
    main()
