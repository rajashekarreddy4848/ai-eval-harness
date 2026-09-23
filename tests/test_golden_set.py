"""Checks on the golden set itself and its helpers. No API calls.

A wrong label makes every suite that reads it wrong, so the data gets tests too.
"""

import json
from collections import Counter

import pytest

from app.knowledge_base import DOCS
from evals.golden import (
    BEHAVIORS, GOLDEN_PATH, answer_cases, decline_cases, is_decline, load_cases,
    missing_facts, normalize,
)

CASES = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
DOC_TITLES = {d["title"] for d in DOCS}
REQUIRED = {"id", "category", "smoke", "expected_behavior", "question", "expected_docs",
            "must_include", "ground_truth"}


def test_ids_are_unique():
    duplicates = [i for i, n in Counter(c["id"] for c in CASES).items() if n > 1]
    assert not duplicates


def test_questions_are_unique():
    duplicates = [q for q, n in Counter(c["question"].lower() for c in CASES).items() if n > 1]
    assert not duplicates


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_is_well_formed(case):
    assert REQUIRED <= case.keys(), f"missing {REQUIRED - case.keys()}"
    assert case["expected_behavior"] in BEHAVIORS
    assert set(case["expected_docs"]) <= DOC_TITLES, "expected_docs must be knowledge-base titles"
    if case["expected_behavior"] == "decline":
        assert not case["expected_docs"] and not case["must_include"]
    else:
        assert case["expected_docs"], "answer cases need the doc(s) that answer them"
        assert case["must_include"] or case.get("rubric"), "answer cases need a check"


@pytest.mark.parametrize("case", [c for c in CASES if c["expected_behavior"] == "answer"],
                         ids=lambda c: c["id"])
def test_facts_appear_in_expected_docs(case):
    """A must_include fact the knowledge base never states would make the case unpassable."""
    source = normalize(" ".join(d["text"] for d in DOCS if d["title"] in case["expected_docs"])).lower()
    derived = case["category"] == "numeric-reasoning" or case["id"] == "REF-04"
    for fact in case["must_include"]:
        if not derived:
            assert normalize(fact).lower() in source, f"{fact!r} is not in {case['expected_docs']}"


def test_every_doc_is_covered():
    covered = {t for c in CASES for t in c["expected_docs"]}
    assert covered == DOC_TITLES


def test_coverage_mix():
    categories = Counter(c["category"] for c in CASES)
    assert len(CASES) >= 30
    assert categories["out-of-scope"] >= 5 and categories["multi-doc"] >= 3


def test_smoke_subset_has_both_behaviors():
    assert answer_cases("smoke") and decline_cases("smoke")
    assert len(load_cases("smoke")) < len(load_cases("full"))


def test_known_issues_are_explained():
    for case in CASES:
        if "known_issue" in case:
            assert isinstance(case["known_issue"], str) and len(case["known_issue"]) > 20, case["id"]


@pytest.mark.parametrize("mode, expected", [("exclude", ["A"]), ("only", ["B"]), ("include", ["A", "B"])])
def test_select_known_issues(monkeypatch, mode, expected):
    from evals.golden import select_known_issues
    monkeypatch.setenv("KNOWN_ISSUES", mode)
    cases = [{"id": "A"}, {"id": "B", "known_issue": "retrieval miss"}]
    assert [c["id"] for c in select_known_issues(cases)] == expected


def test_unknown_known_issues_mode_is_rejected(monkeypatch):
    from evals.golden import select_known_issues
    monkeypatch.setenv("KNOWN_ISSUES", "maybe")
    with pytest.raises(ValueError):
        select_known_issues([])


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError):
        load_cases("everything")


@pytest.mark.parametrize("raw, expected", [
    ("30\u202fdays", "30 days"),
    ("I don\u2019t know", "I don't know"),
    ("5\u20137 business days", "5-7 business days"),
    ("2\u20113 days", "2-3 days"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_is_decline():
    assert is_decline("I don\u2019t know.")
    assert is_decline("That information is not available in the FAQ.")
    assert is_decline("I\u2019m sorry, but I can\u2019t create or provide discount codes.")
    assert is_decline("I can't access or share personal account information.")
    assert not is_decline("You have 30 days to request a refund.")
    # "can't" about the user, not the bot, is not a refusal
    assert not is_decline("Free plan users can't export JSON; Pro costs $12/month.")


def test_missing_facts_ignores_typography():
    case = {"must_include": ["5-7", "$15"]}
    assert missing_facts("It takes 5\u20137 days and costs $15.", case) == []
    assert missing_facts("It takes a week.", case) == ["5-7", "$15"]
