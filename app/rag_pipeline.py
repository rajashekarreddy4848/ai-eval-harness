"""Minimal RAG pipeline: TF-IDF retrieval + pluggable LLM generation.

Kept deliberately simple (no vector DB, no embeddings API) so the project
runs cheaply and the focus stays on the EVAL/TEST harness, not the app itself.
Supports Groq/Gemini (free tiers) and Anthropic (paid) as providers.
"""

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.knowledge_base import DOCS

_vectorizer = TfidfVectorizer()
_doc_texts = [d["text"] for d in DOCS]
_doc_matrix = _vectorizer.fit_transform(_doc_texts)

SYSTEM_PROMPT = (
    "You are a support assistant. Answer the user's question using ONLY the "
    "provided context. If the context doesn't contain the answer, say you "
    "don't know rather than guessing. Be concise (2-3 sentences max)."
)


def retrieve(query: str, k: int = 2) -> list[dict]:
    """Return the top-k most relevant docs for the query via TF-IDF cosine similarity."""
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_matrix).flatten()
    top_indices = scores.argsort()[::-1][:k]
    return [DOCS[i] for i in top_indices]


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in contexts)
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the context above."
    )


def generate_answer(query: str) -> dict:
    """Retrieve context and call the LLM. Returns answer + contexts used (for eval)."""
    contexts = retrieve(query)
    user_prompt = build_prompt(query, contexts)

    answer = _call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "query": query,
        "contexts": [c["text"] for c in contexts],
        "answer": answer,
    }


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls whichever LLM provider has a key configured.

    Checked in order: GROQ_API_KEY (free tier), GEMINI_API_KEY (free tier),
    ANTHROPIC_API_KEY (paid). Set LLM_PROVIDER explicitly to force one.
    """
    provider = os.environ.get("LLM_PROVIDER")

    if provider == "groq" or (provider is None and os.environ.get("GROQ_API_KEY")):
        return _call_groq(system_prompt, user_prompt)
    if provider == "gemini" or (provider is None and os.environ.get("GEMINI_API_KEY")):
        return _call_gemini(system_prompt, user_prompt)
    if provider == "anthropic" or (provider is None and os.environ.get("ANTHROPIC_API_KEY")):
        return _call_anthropic(system_prompt, user_prompt)

    raise RuntimeError(
        "No LLM API key found. Set one of GROQ_API_KEY, GEMINI_API_KEY, or "
        "ANTHROPIC_API_KEY in your environment."
    )


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Free tier: https://console.groq.com — no card required."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Free tier: https://aistudio.google.com/apikey — no card required."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, max_output_tokens=300
        ),
    )
    return response.text


def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Requires paid credits: https://console.anthropic.com/settings/billing"""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    result = generate_answer("How long do I have to get a refund?")
    print(result)
