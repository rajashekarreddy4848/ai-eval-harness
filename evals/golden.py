"""The golden eval set shared by every suite (retrieval, deepeval, ragas, promptfoo).

Each case in golden_set.json says what a correct response looks like:
  expected_behavior  "answer", or "decline" when the FAQ doesn't cover the question
  expected_docs      titles of the knowledge-base docs retrieval should return
  must_include       facts a correct answer has to contain (checked after normalize())
  ground_truth       reference answer, used by ragas context recall
  rubric             optional extra check for an LLM grader (false-premise cases)
  smoke              part of the small subset the slow LLM-graded suites run by default

EVAL_SUBSET=smoke (default) or EVAL_SUBSET=full picks how many cases the
LLM-graded suites run. Free-tier judges allow only a few calls per minute.
"""

import json
import os
import unicodedata
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"
BEHAVIORS = {"answer", "decline"}

# How the bot says it can't answer. Checked against normalize()d, lowercased text.
DECLINE_PHRASES = [
    "don't know", "do not know", "not available", "no information",
    "not mentioned", "doesn't mention", "does not mention", "not covered",
    "doesn't cover", "does not cover", "can't help", "cannot help", "unable to",
]

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
    return any(phrase in text for phrase in DECLINE_PHRASES)


def missing_facts(answer: str, case: dict) -> list[str]:
    text = normalize(answer).lower()
    return [fact for fact in case["must_include"] if normalize(fact).lower() not in text]


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
