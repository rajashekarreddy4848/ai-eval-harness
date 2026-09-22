"""LLM output quality tests using deepeval — run like any pytest suite.

Run with: pytest tests/test_deepeval.py

Each test pulls a real answer from the RAG pipeline and scores it against
deepeval's built-in metrics (hallucination, answer relevancy, faithfulness).
Thresholds are intentionally strict-ish to catch regressions.
"""

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

EVAL_MODEL = get_eval_model()

TEST_QUERIES = [
    "How long do I have to request a refund?",
    "How long does express shipping take?",
    "What happens after I delete my account?",
    "How much does the Team plan cost per user?",
]


def _build_test_case(query: str) -> LLMTestCase:
    result = generate_answer(query)
    return LLMTestCase(
        input=query,
        actual_output=result["answer"],
        retrieval_context=result["contexts"],
        context=result["contexts"],
    )


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_answer_relevancy(query):
    test_case = _build_test_case(query)
    metric = AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_faithfulness_to_context(query):
    """Answer must not contradict or invent facts beyond the retrieved context."""
    test_case = _build_test_case(query)
    metric = FaithfulnessMetric(threshold=0.7, model=EVAL_MODEL)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_no_hallucination(query):
    test_case = _build_test_case(query)
    # Since deepeval 1.x+: threshold is now a MINIMUM passing score (1 = perfectly
    # faithful, 0 = fully hallucinated) -- not a max violation rate like before.
    metric = HallucinationMetric(threshold=0.7, model=EVAL_MODEL)
    assert_test(test_case, [metric])


def test_out_of_scope_question_is_handled_gracefully():
    """Regression guard: asking something outside the KB shouldn't produce a confident lie."""
    result = generate_answer("What's the weather like in Hyderabad today?")
    # Normalize curly apostrophes (’) to straight ones (') -- LLMs often use the former.
    lower = result["answer"].lower().replace("’", "'")
    assert any(
        phrase in lower for phrase in ["don't know", "not available", "cannot", "no information"]
    ), f"Expected a graceful 'I don't know', got: {result['answer']}"
