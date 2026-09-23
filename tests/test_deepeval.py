"""LLM output quality tests using deepeval — run like any pytest suite.

Run with: pytest tests/test_deepeval.py

Each test pulls a real answer from the RAG pipeline and scores it against
deepeval's built-in metrics (hallucination, answer relevancy, faithfulness).
Thresholds are intentionally strict-ish to catch regressions.
"""

from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from app.eval_model import get_eval_model
from app.rag_pipeline import generate_answer

TEST_QUERIES = [
    "How long do I have to request a refund?",
    "How long does express shipping take?",
    "What happens after I delete my account?",
    "How much does the Team plan cost per user?",
]


@lru_cache(maxsize=None)
def _answer(query: str) -> dict:
    """Generate each answer once, so every metric grades the same answer (and we pay once)."""
    return generate_answer(query)


@pytest.fixture(scope="session")
def judge():
    return get_eval_model()


def _build_test_case(query: str) -> LLMTestCase:
    result = _answer(query)
    return LLMTestCase(
        input=query,
        actual_output=result["answer"],
        retrieval_context=result["contexts"],
        context=result["contexts"],
    )


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_answer_relevancy(query, judge):
    test_case = _build_test_case(query)
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_faithfulness_to_context(query, judge):
    """Answer must not contradict or invent facts beyond the retrieved context."""
    test_case = _build_test_case(query)
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_no_hallucination(query, judge):
    test_case = _build_test_case(query)
    # In deepeval 4.x the hallucination score is the share of contexts the answer
    # agrees with (1 = no hallucination), and a test passes when score >= threshold.
    # Older 1.x releases scored the opposite way, so requirements.txt pins deepeval 4.x.
    metric = HallucinationMetric(threshold=0.7, model=judge)
    assert_test(test_case, [metric])


def test_out_of_scope_question_is_handled_gracefully():
    """Regression guard: asking something outside the KB shouldn't produce a confident lie."""
    result = _answer("What's the weather like in Hyderabad today?")
    # Normalize curly apostrophes (’) to straight ones (') -- LLMs often use the former.
    lower = result["answer"].lower().replace("’", "'")
    assert any(
        phrase in lower for phrase in ["don't know", "not available", "cannot", "no information"]
    ), f"Expected a graceful 'I don't know', got: {result['answer']}"
