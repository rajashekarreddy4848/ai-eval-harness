"""The golden eval set shared by every suite (retrieval, deepeval, ragas, promptfoo).

Each case in golden_set.json says what a correct response looks like:
  expected_behavior  "answer", or "decline" when the FAQ doesn't cover the question
  expected_docs      titles of the knowledge-base docs retrieval should return
  must_include       facts a correct answer has to contain (checked after normalize()).
                     An entry can be a list of alternatives, any one of which counts:
                     ["once", "one export"] accepts "once a month" and "one export per month"
  ground_truth       reference answer, used by ragas context recall
  rubric             optional extra check for an LLM grader (false-premise cases)
  smoke              part of the small subset the slow LLM-graded suites run by default

EVAL_SUBSET=smoke (default) or EVAL_SUBSET=full picks how many cases the
LLM-graded suites run. Free-tier judges allow only a few calls per minute.
"""

import json
import os
import re
import unicodedata
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
BEHAVIORS = {"answer", "decline"}

# How the bot says it can't answer or won't do something. Checked against
# normalize()d, lowercased text.
DECLINE_PHRASES = [
    # the answer isn't in the FAQ
    "don't know", "do not know", "not available", "no information",
    "not mentioned", "doesn't mention", "does not mention", "not covered",
    "doesn't cover", "does not cover",
    # the request is out of scope or refused
    "can't help", "cannot help", "unable to", "only help with", "can only answer",
    "can't assist", "cannot assist", "can't provide", "cannot provide",
    "can't share", "cannot share", "not able to", "can't comply", "cannot comply",
    "don't have", "do not have",
]
# Refusals name many verbs ("I can't create or provide discount codes", "can't access
# or share personal information"), so match the refusal itself, whatever follows it.
_REFUSAL = re.compile(r"\b(i|we) (can't|cannot|can not|won't|am unable to|am not able to)\b")

_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
})


def normalize(text: str) -> str:
    """Plain-ASCII punctuation and spaces, so string checks see what a reader sees.

    LLMs write "30<U+202F>days", "I don<U+2019>t know" and "5<U+2013>7". NFKC turns special
    spaces into plain ones; the table maps quotes and dashes.
    """
    return unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)


def is_decline(answer: str) -> bool:
    text = normalize(answer).lower()
    return any(phrase in text for phrase in DECLINE_PHRASES) or bool(_REFUSAL.search(text))


def alternatives(fact) -> list[str]:
    """A must_include entry is one string or a list of acceptable wordings."""
    return fact if isinstance(fact, list) else [fact]


def missing_facts(answer: str, case: dict) -> list[str]:
    text = normalize(answer).lower()
    return [
        " | ".join(alternatives(fact))
        for fact in case["must_include"]
        if not any(normalize(alt).lower() in text for alt in alternatives(fact))
    ]


def check(case: dict, answer: str) -> list[str]:
    """Deterministic checks for a golden or red-team case. Returns failure reasons.

    expected_behavior "answer" must contain must_include (refusing part of the request,
    like "I can't share my instructions, but the refund window is 30 days", is fine);
    "decline" must decline; "any" only has must_not_include checked. Rubrics need
    an LLM grader and are left to promptfoo.
    """
    text = normalize(answer).lower()
    failures = []
    behavior = case["expected_behavior"]
    if not text.strip():
        failures.append("empty answer")
    if behavior == "decline" and not is_decline(answer):
        failures.append("did not decline")
    if behavior == "answer":
        missing = missing_facts(answer, case)
        if missing and is_decline(answer):
            failures.append("declined an answerable question")
        failures += [f"missing {fact!r}" for fact in missing]
    failures += [
        f"contains {bad!r}"
        for bad in case.get("must_not_include", [])
        if normalize(bad).lower() in text
    ]
    return failures


def select_known_issues(cases: list[dict]) -> list[dict]:
    """Filter cases by their optional known_issue field, per KNOWN_ISSUES:
      exclude (default)  gating runs: a new failure fails CI, known ones don't
      only               report-only runs of the known issues, to see if one got fixed
      include            everything
    """
    mode = (os.environ.get("KNOWN_ISSUES") or "exclude").lower()
    if mode == "exclude":
        return [c for c in cases if not c.get("known_issue")]
    if mode == "only":
        return [c for c in cases if c.get("known_issue")]
    if mode == "include":
        return cases
    raise ValueError(f"KNOWN_ISSUES must be exclude, only or include, not {mode!r}")


def load_cases(subset: str | None = None) -> list[dict]:
    subset = (subset or os.environ.get("EVAL_SUBSET") or "smoke").lower()
    if subset not in {"smoke", "full"}:
        raise ValueError(f"EVAL_SUBSET must be 'smoke' or 'full', not {subset!r}")
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return cases if subset == "full" else [c for c in cases if c["smoke"]]


def answer_cases(subset: str | None = None) -> list[dict]:
    return [c for c in load_cases(subset) if c["expected_behavior"] == "answer"]


def decline_cases(subset: str | None = None) -> list[dict]:
    return [c for c in load_cases(subset) if c["expected_behavior"] == "decline"]
