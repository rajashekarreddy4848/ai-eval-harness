"""LLM output quality tests using deepeval — run like any pytest suite.

Run with: pytest tests/test_deepeval.py            (smoke subset of the golden set)
          EVAL_SUBSET=full pytest tests/test_deepeval.py

Cases come from evals/golden_set.json. Answerable questions are scored with
deepeval's LLM-graded metrics (answer relevancy, faithfulness, hallucination,
correctness); questions the FAQ doesn't cover must get a "don't know" instead
of a guess. Thresholds are intentionally strict-ish to catch regressions.

Why correctness is here: relevancy, faithfulness and hallucination never see
the right answer. When retrieval misses the pricing doc, "I don't know." to
"How much does the Pro plan cost?" scores 1.0 on relevancy AND faithfulness.
Only a check against the expected answer catches that.
"""

from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.eval_model import get_eval_model
from app.rag_pipeline import generate_answer
from evals.golden import answer_cases, decline_cases, is_decline

ANSWER_CASES = [pytest.param(c, id=c["id"]) for c in answer_cases()]
DECLINE_CASES = [pytest.param(c, id=c["id"]) for c in decline_cases()]


@lru_cache(maxsize=None)
def _answer(query: str) -> dict:
    """Generate each answer once, so every metric grades the same answer (and we pay once)."""
    return generate_answer(query)


@pytest.fixture(scope="session")
def judge():
    return get_eval_model()


def _build_test_case(case: dict) -> LLMTestCase:
    result = _answer(case["question"])
    return LLMTestCase(
        input=case["question"],
        actual_output=result["answer"],
        expected_output=case["ground_truth"],
        retrieval_context=result["contexts"],
        context=result["contexts"],
    )


@pytest.mark.parametrize("case", ANSWER_CASES)
def test_answer_relevancy(case, judge):
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    assert_test(_build_test_case(case), [metric])


@pytest.mark.parametrize("case", ANSWER_CASES)
def test_faithfulness_to_context(case, judge):
    """Answer must not contradict or invent facts beyond the retrieved context."""
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    assert_test(_build_test_case(case), [metric])


@pytest.mark.parametrize("case", ANSWER_CASES)
def test_no_hallucination(case, judge):
    # In deepeval 4.x the hallucination score is the share of contexts the answer
    # agrees with (1 = no hallucination), and a test passes when score >= threshold.
    # Older 1.x releases scored the opposite way, so requirements.txt pins deepeval 4.x.
    metric = HallucinationMetric(threshold=0.7, model=judge)
    assert_test(_build_test_case(case), [metric])


@pytest.mark.parametrize("case", ANSWER_CASES)
def test_correctness_against_ground_truth(case, judge):
    metric = GEval(
        name="Correctness",
        criteria=(
            "Does the actual output give the key facts in the expected output, without "
            "contradicting them? Extra wording is fine. Saying 'I don't know', or leaving "
            "out a key fact, when the expected output contains an answer is incorrect."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=judge,
    )
    assert_test(_build_test_case(case), [metric])


@pytest.mark.parametrize("case", DECLINE_CASES)
def test_out_of_scope_question_is_declined(case):
    """Regression guard: a question the FAQ doesn't cover must not get a confident guess."""
    answer = _answer(case["question"])["answer"]
    assert is_decline(answer), f"Expected a graceful 'I don't know', got: {answer}"
