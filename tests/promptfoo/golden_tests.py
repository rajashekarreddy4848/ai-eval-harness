"""Builds promptfoo test cases from evals/golden_set.json.

promptfooconfig.yaml loads these with tests: file://golden_tests.py:generate_tests.
Promptfoo runs every golden case (not just the smoke subset): each case is one
generation plus at most one grader call, so the full set is cheap enough.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.golden import load_cases  # noqa: E402

DECLINE_RUBRIC = (
    "The answer says it doesn't know or that the information isn't available, "
    "instead of answering or making something up."
)

# Non-functional checks that aren't about a specific fact.
EXTRA_TESTS = [
    {
        "description": "NFR: answer stays concise and fast",
        "vars": {"prompt": "What are the shipping options?"},
        "assert": [
            {"type": "latency", "threshold": 8000},
            {"type": "llm-rubric", "value": "Answer is 3 sentences or fewer"},
        ],
    },
]


def _asserts(case: dict) -> list[dict]:
    if case["expected_behavior"] == "decline":
        return [{"type": "llm-rubric", "value": DECLINE_RUBRIC}]
    checks = [{"type": "icontains", "value": fact} for fact in case["must_include"]]
    # A correct answer must not be a refusal (catches retrieval misses).
    checks.append({"type": "not-icontains", "value": "I don't know"})
    if case.get("rubric"):
        checks.append({"type": "llm-rubric", "value": case["rubric"]})
    return checks


def generate_tests(config=None):
    tests = [
        {
            "description": f"{case['id']} [{case['category']}] {case['question']}",
            "vars": {"prompt": case["question"]},
            "assert": _asserts(case),
            "metadata": {"id": case["id"], "category": case["category"]},
        }
        for case in load_cases("full")
    ]
    return tests + EXTRA_TESTS
