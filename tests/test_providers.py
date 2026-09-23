"""Unit tests for generator/judge selection. No API calls, no keys needed."""

import pytest

from app.providers import PROVIDERS, generator_provider, is_self_graded, judge_provider

ENV_VARS = ["LLM_PROVIDER", "JUDGE_PROVIDER", "JUDGE_MODEL"] + [
    var for p in PROVIDERS.values() for var in (p.key_env, p.model_env)
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def keys(monkeypatch, *names):
    for name in names:
        monkeypatch.setenv(PROVIDERS[name].key_env, "test-key")


def test_no_keys_raises():
    with pytest.raises(RuntimeError, match="No LLM API key"):
        generator_provider()


def test_generator_prefers_free_tier(monkeypatch):
    keys(monkeypatch, "anthropic", "gemini", "groq")
    assert generator_provider().name == "groq"


def test_judge_differs_from_generator(monkeypatch):
    keys(monkeypatch, "groq", "gemini")
    assert generator_provider().name == "groq"
    assert judge_provider().name == "gemini"
    assert not is_self_graded()


def test_forced_generator_moves_judge(monkeypatch):
    keys(monkeypatch, "groq", "gemini")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert generator_provider().name == "gemini"
    assert judge_provider().name == "groq"


def test_single_provider_warns_about_self_grading(monkeypatch):
    keys(monkeypatch, "groq")
    with pytest.warns(UserWarning, match="self-grading"):
        assert judge_provider().name == "groq"


def test_forced_judge_is_respected(monkeypatch):
    keys(monkeypatch, "groq", "gemini", "anthropic")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    assert judge_provider().name == "anthropic"


def test_forcing_judge_to_generator_is_self_graded(monkeypatch):
    keys(monkeypatch, "groq")
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    with pytest.warns(UserWarning, match="self-grading"):
        assert is_self_graded()
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")
    with pytest.warns(UserWarning, match="self-grading"):
        assert is_self_graded()  # both roles read GROQ_MODEL, so it is still the same model


def test_judge_model_on_same_provider_is_not_self_graded(monkeypatch, recwarn):
    keys(monkeypatch, "groq")
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "qwen/qwen3.8-27b")
    judge = judge_provider()
    assert (judge.name, judge.model) == ("groq", "qwen/qwen3.8-27b")
    assert generator_provider().model == "openai/gpt-oss-120b"
    assert not is_self_graded()
    assert not [w for w in recwarn if "self-grading" in str(w.message)]


def test_judge_model_overrides_other_provider_default(monkeypatch):
    keys(monkeypatch, "groq", "gemini")
    monkeypatch.setenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
    assert judge_provider().model == "gemini-3.5-flash-lite"


def test_unknown_provider_name(monkeypatch):
    keys(monkeypatch, "groq")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="not one of"):
        generator_provider()


def test_forced_provider_without_key(monkeypatch):
    keys(monkeypatch, "groq")
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        judge_provider()


def test_model_override(monkeypatch):
    keys(monkeypatch, "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    assert generator_provider().model == "gemini-custom"


def test_provider_names_are_case_insensitive(monkeypatch):
    keys(monkeypatch, "groq", "gemini")
    monkeypatch.setenv("LLM_PROVIDER", " Gemini ")
    assert generator_provider().name == "gemini"
