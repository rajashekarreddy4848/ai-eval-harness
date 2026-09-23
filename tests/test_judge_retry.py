"""Unit tests for how the judge handles rate limits and outages. No API calls."""

from types import SimpleNamespace

import httpx
import openai
import pytest

import app.eval_model as eval_model
from app.eval_model import JudgeQuotaExhausted, OpenAICompatibleJudge, _retry_delay
from app.providers import PROVIDERS

_REQUEST = httpx.Request("POST", "https://judge.example/chat/completions")
PER_MINUTE = "Quota exceeded ... quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 2s."
PER_DAY = "Quota exceeded ... quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 57s."


def rate_limited(message):
    return openai.RateLimitError(message, response=httpx.Response(429, request=_REQUEST), body=None)


def unavailable():
    return openai.InternalServerError("503 UNAVAILABLE", response=httpx.Response(503, request=_REQUEST), body=None)


class FakeClient:
    """Raises the queued errors in order, then answers."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 1}'))])


@pytest.fixture
def judge(monkeypatch):
    sleeps = []
    monkeypatch.setattr(eval_model.time, "sleep", sleeps.append)
    j = OpenAICompatibleJudge.__new__(OpenAICompatibleJudge)
    j.provider = PROVIDERS["gemini"]
    j.sleeps = sleeps
    return j


def test_per_minute_limit_is_retried(judge):
    judge.model = FakeClient([rate_limited(PER_MINUTE), rate_limited(PER_MINUTE)])
    assert judge.generate("grade this") == '{"score": 1}'
    assert judge.model.calls == 3 and judge.sleeps == [3.0, 3.0]


def test_daily_quota_fails_at_once_without_waiting(judge):
    judge.model = FakeClient([rate_limited(PER_DAY)])
    with pytest.raises(JudgeQuotaExhausted, match="daily free-tier quota"):
        judge.generate("grade this")
    assert judge.model.calls == 1 and judge.sleeps == []


def test_server_unavailable_is_retried(judge):
    judge.model = FakeClient([unavailable()])
    assert judge.generate("grade this") == '{"score": 1}'
    assert judge.sleeps == [5]


def test_sdk_retries_are_disabled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = OpenAICompatibleJudge(PROVIDERS["gemini"]).load_model()
    assert client.max_retries == 0


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
