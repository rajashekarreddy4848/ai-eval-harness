"""Custom deepeval judge model, so metrics don't require a separate OPENAI_API_KEY.

deepeval's built-in metrics (AnswerRelevancyMetric, FaithfulnessMetric, etc.)
default to grading with OpenAI. This wraps whichever provider we already
configured in app/rag_pipeline.py so the SAME free-tier key judges the answers.
"""

import os

from deepeval.models import DeepEvalBaseLLM


class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


def get_eval_model():
    """Returns a judge model matching whichever provider key is configured."""
    if os.environ.get("GROQ_API_KEY"):
        return GroqEvalModel()
    # No wrapper needed for Anthropic/OpenAI -- deepeval supports those directly
    # by name; returning None lets metrics fall back to their own default.
    return None
