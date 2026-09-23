"""LLM provider config shared by the app (the generator) and the eval suites (the judge).

The generator answers questions; the judge grades those answers. They should
be different models: a model grading its own output tends to score it too
kindly (self-grading bias). Picking both from one table keeps that choice in
one place instead of scattered across the app, deepeval, ragas and promptfoo.

Environment variables:
  GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY   which providers are available
  LLM_PROVIDER    force the generator (groq | gemini | anthropic)
  JUDGE_PROVIDER  force the judge; default is the first available provider
                  that is NOT the generator
  JUDGE_MODEL     judge model, e.g. a different model family on the same provider
  GROQ_MODEL / GEMINI_MODEL / ANTHROPIC_MODEL         override a default model
"""

import os
import warnings
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str
    model_env: str
    default_model: str
    base_url: str  # OpenAI-compatible endpoint, used by the judge clients

    @property
    def api_key(self) -> str:
        return os.environ[self.key_env]

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env) or self.default_model

    def configured(self) -> bool:
        return bool(os.environ.get(self.key_env))


PROVIDERS = {
    "groq": Provider(
        "groq", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b",
        "https://api.groq.com/openai/v1",
    ),
    "gemini": Provider(
        "gemini", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash-lite",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "anthropic": Provider(
        "anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5",
        "https://api.anthropic.com/v1/",
    ),
}
# Free tiers first.
ORDER = ("groq", "gemini", "anthropic")


def _forced(env_var: str) -> Provider:
    name = os.environ[env_var].strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"{env_var}={name!r} is not one of {', '.join(PROVIDERS)}.")
    provider = PROVIDERS[name]
    if not provider.configured():
        raise RuntimeError(f"{env_var}={name} but {provider.key_env} is not set.")
    return provider


def generator_provider() -> Provider:
    """The provider that answers user questions."""
    if os.environ.get("LLM_PROVIDER"):
        return _forced("LLM_PROVIDER")
    for name in ORDER:
        if PROVIDERS[name].configured():
            return PROVIDERS[name]
    raise RuntimeError(
        "No LLM API key found. Set one of GROQ_API_KEY, GEMINI_API_KEY, or "
        "ANTHROPIC_API_KEY in your environment or .env file."
    )


def judge_provider() -> Provider:
    """The provider that grades answers. Prefers a different model from the generator."""
    generator = generator_provider()
    if os.environ.get("JUDGE_PROVIDER"):
        judge = _forced("JUDGE_PROVIDER")
    else:
        others = [n for n in ORDER if n != generator.name and PROVIDERS[n].configured()]
        judge = PROVIDERS[others[0]] if others else generator
    if os.environ.get("JUDGE_MODEL"):
        judge = replace(judge, model_env="JUDGE_MODEL")
    if judge.name == generator.name and judge.model == generator.model:
        warnings.warn(
            f"{generator.name}:{generator.model} is grading its own answers "
            "(self-grading bias). Set JUDGE_MODEL, or add a second provider key.",
            stacklevel=2,
        )
    return judge


def is_self_graded() -> bool:
    judge = judge_provider()
    generator = generator_provider()
    return judge.name == generator.name and judge.model == generator.model
