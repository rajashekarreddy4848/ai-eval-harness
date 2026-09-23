"""Unit tests for how long the judge waits after a rate-limit error. No API calls."""

from types import SimpleNamespace

from app.eval_model import _retry_delay


def error(message="", headers=None):
    err = Exception(message)
    err.response = SimpleNamespace(headers=headers or {})
    return err


def test_uses_retry_after_header():
    assert _retry_delay(error(headers={"retry-after": "7"}), attempt=0) == 8


def test_parses_groq_message_in_seconds():
    assert _retry_delay(error("Please try again in 12.5s."), attempt=0) == 13.5


def test_parses_milliseconds():
    assert _retry_delay(error("Please try again in 500ms."), attempt=0) == 1.5


def test_parses_gemini_message():
    assert _retry_delay(error("Please retry in 29.5s."), attempt=0) == 30.5


def test_falls_back_to_capped_exponential_backoff():
    assert _retry_delay(error("rate limited"), attempt=0) == 5
    assert _retry_delay(error("rate limited"), attempt=2) == 20
    assert _retry_delay(error("rate limited"), attempt=10) == 60
