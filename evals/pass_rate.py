"""Run each case several times and report how often it passes.

LLM answers vary between runs, so a single pass or fail can be luck: the golden
set's poem request (OOS-05) passed once and then failed 6 times out of 6.
This script only calls the generator and uses deterministic checks
(evals/golden.py check()), so it needs no judge.

Run with: python evals/pass_rate.py                       # red-team set, 3 runs each
          python evals/pass_rate.py --set golden --repeat 3
          PROMPT_VERSION=v1 python evals/pass_rate.py      # a specific system prompt
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag_pipeline import DEFAULT_PROMPT_VERSION, generate_answer  # noqa: E402
from evals.golden import check, load_cases  # noqa: E402

REDTEAM_PATH = ROOT / "evals" / "redteam_set.json"
RESULTS_DIR = ROOT / "evals" / "results"


def load(name: str) -> list[dict]:
    if name == "redteam":
        return json.loads(REDTEAM_PATH.read_text(encoding="utf-8"))
    return load_cases("full")


def run(cases: list[dict], repeat: int) -> list[dict]:
    rows = []
    for case in cases:
        runs = []
        for _ in range(repeat):
            answer = generate_answer(case["question"], injected_docs=[case["inject_doc"]] if case.get("inject_doc") else None)["answer"]
            runs.append({"answer": answer, "failures": check(case, answer)})
        passes = sum(not r["failures"] for r in runs)
        rows.append({"id": case["id"], "category": case["category"], "holdout": case.get("holdout", False),
                     "passes": passes, "runs": runs})
        first_fail = next((r for r in runs if r["failures"]), None)
        note = f"  <- {first_fail['failures'][0]}: {first_fail['answer'][:80]!r}" if first_fail else ""
        print(f"{case['id']:<10} {passes}/{repeat}{note}", flush=True)
    return rows


def rescore(rows: list[dict], cases: list[dict]) -> list[dict]:
    """Re-check saved answers with the current check() and case definitions. No API calls."""
    by_id = {c["id"]: c for c in cases}
    for row in rows:
        for run_ in row["runs"]:
            run_["failures"] = check(by_id[row["id"]], run_["answer"])
        row["passes"] = sum(not r["failures"] for r in row["runs"])
        first_fail = next((r for r in row["runs"] if r["failures"]), None)
        note = f"  <- {first_fail['failures'][0]}: {first_fail['answer'][:80]!r}" if first_fail else ""
        print(f"{row['id']:<10} {row['passes']}/{len(row['runs'])}{note}")
    return rows


def summarize(rows: list[dict], repeat: int) -> None:
    def line(label, group):
        passed = sum(r["passes"] for r in group)
        always = sum(r["passes"] == repeat for r in group)
        print(f"{label:<22} runs passed {passed}/{len(group) * repeat} ({passed / (len(group) * repeat):.0%})"
              f" | cases passing every run {always}/{len(group)}")

    print()
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    for category, group in sorted(by_category.items()):
        line(category, group)
    print()
    if any(r["holdout"] for r in rows):
        line("tuning cases", [r for r in rows if not r["holdout"]])
        line("held-out cases", [r for r in rows if r["holdout"]])
    line("ALL", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["redteam", "golden"], default="redteam")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rescore", metavar="RESULTS_JSON", help="re-check saved answers, no API calls")
    args = parser.parse_args()

    if args.rescore:
        rows = rescore(json.loads(Path(args.rescore).read_text(encoding="utf-8")), load(args.set))
        summarize(rows, len(rows[0]["runs"]))
        return

    version = os.environ.get("PROMPT_VERSION") or DEFAULT_PROMPT_VERSION
    print(f"set={args.set} repeat={args.repeat} prompt={version}\n")
    rows = run(load(args.set), args.repeat)
    summarize(rows, args.repeat)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"pass_rate_{args.set}_{version}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
