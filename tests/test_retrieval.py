"""Retrieval tests: does TF-IDF return the doc(s) that answer each golden question?

No LLM involved, so it's free, fast and deterministic, and it runs on every PR.
When an answer is wrong, this separates "retrieval missed the doc" from "the
model misused a doc it was given".
"""

import pytest

from app.rag_pipeline import retrieve
from evals.golden import answer_cases

CASES = answer_cases("full")

# Known retrieval misses with the hybrid retriever (the word-only retriever missed
# 7; see README). Both need meaning, not spelling, so lexical retrieval can't fix them.
# strict=True: once a fix makes one pass, pytest fails until its entry is removed.
KNOWN_MISSES = {
    "REF-04": "synonym: 'money back' never appears; the doc says 'refund'",
    "FPR-03": "'Enterprise' and 'sign up' appear in no doc; 'plan' alone matches Data Export best",
}


def _param(case):
    marks = [pytest.mark.xfail(reason=KNOWN_MISSES[case["id"]], strict=True)] if case["id"] in KNOWN_MISSES else []
    return pytest.param(case, id=case["id"], marks=marks)


@pytest.mark.parametrize("case", [_param(c) for c in CASES])
def test_expected_docs_are_retrieved(case):
    retrieved = [d["title"] for d in retrieve(case["question"])]
    missing = [t for t in case["expected_docs"] if t not in retrieved]
    assert not missing, f"missing {missing}; retrieved {retrieved}"
