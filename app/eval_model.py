"""Judge models for deepeval and ragas.

Both libraries grade with OpenAI by default, which would need an OPENAI_API_KEY
this project doesn't use. These helpers point them at the judge provider from
app/providers.py instead, which is a different model from the generator
whenever a second provider key is available.
"""

import os
import re
import time

from deepeval.models import DeepEvalBaseLLM

from app.providers import Provider, judge_provider

# Free tiers allow only a few judge calls per minute, so a rate-limit error means
# "wait", not "the answer failed". Retry it instead of reporting a failed test.
# A per-DAY quota is different: waiting a minute won't help, so fail at once.
MAX_RATE_LIMIT_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "8"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "900"))


class JudgeQuotaExhausted(RuntimeError):
    """The judge's daily quota is used up: an infrastructure problem, not a failed answer."""


def _is_daily_quota(error) -> bool:
    # Gemini names the quota, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier.
    return "PerDay" in str(error)


def _retry_delay(error, attempt: int) -> float:
    """Seconds to wait before retrying, taken from the provider's error when it says."""
    header = getattr(getattr(error, "response", None), "headers", {}).get("retry-after")
    if header:
        return float(header) + 1
    match = re.search(r"(?:try again|retry) in ([\d.]+)\s*(ms|s)", str(error), re.IGNORECASE)
    if match:
        seconds = float(match.group(1)) / (1000 if match.group(2).lower() == "ms" else 1)
        return seconds + 1
    return min(5 * 2**attempt, 60)


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """deepeval judge for any provider with an OpenAI-compatible endpoint (Groq, Gemini)."""

    def __init__(self, provider: Provider):
        self.provider = provider
        super().__init__(provider.model)

    def load_model(self):
        from openai import OpenAI

        # max_retries=0: generate() does the retrying. The SDK's own retries stacked
        # on top turned one exhausted quota into ~15 minutes of waiting per test in CI.
        return OpenAI(api_key=self.provider.api_key, base_url=self.provider.base_url, max_retries=0)

    def generate(self, prompt: str) -> str:
        from openai import APIConnectionError, InternalServerError, RateLimitError

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.model.chat.completions.create(
                    model=self.provider.model,
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except RateLimitError as error:
                if _is_daily_quota(error):
                    raise JudgeQuotaExhausted(
                        f"{self.get_model_name()} has used its daily free-tier quota; waiting "
                        "won't help. Gemini resets at midnight Pacific time, or set JUDGE_MODEL "
                        "to another model."
                    ) from error
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(_retry_delay(error, attempt))
            except (InternalServerError, APIConnectionError):
                # 5xx or network trouble ("503 service unavailable"): usually temporary.
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(5 * 2**attempt, 60))

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"{self.provider.name}:{self.provider.model}"


def get_eval_model():
    """deepeval judge model for the configured judge provider."""
    judge = judge_provider()
    if judge.name == "anthropic":
        from deepeval.models import AnthropicModel

        return AnthropicModel(model=judge.model, api_key=judge.api_key, temperature=0)
    return OpenAICompatibleJudge(judge)


def get_ragas_llm():
    """ragas judge LLM, via the judge provider's OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    judge = judge_provider()
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=judge.model,
            api_key=judge.api_key,
            base_url=judge.base_url,
            temperature=0,
            max_tokens=JUDGE_MAX_TOKENS,
            max_retries=3,  # langchain retries 429s too, so keep a daily-quota failure short
        )
    )
