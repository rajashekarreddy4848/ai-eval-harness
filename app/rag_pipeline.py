"""Minimal RAG pipeline: TF-IDF retrieval + pluggable LLM generation.

Kept deliberately simple (no vector DB, no embeddings API) so the project
runs cheaply and the focus stays on the EVAL/TEST harness, not the app itself.
Supports Groq/Gemini (free tiers) and Anthropic (paid) as providers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.knowledge_base import DOCS
from app.providers import PROVIDERS, generator_provider

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
    """Calls the generator provider (see app/providers.py for how it is chosen)."""
    calls = {"groq": _call_groq, "gemini": _call_gemini, "anthropic": _call_anthropic}
    return calls[generator_provider().name](system_prompt, user_prompt)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Free tier: https://console.groq.com — no card required."""
    from openai import OpenAI

    groq = PROVIDERS["groq"]
    client = OpenAI(api_key=groq.api_key, base_url=groq.base_url)
    response = client.chat.completions.create(
        model=groq.model,
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

    gemini = PROVIDERS["gemini"]
    client = genai.Client(api_key=gemini.api_key)
    response = client.models.generate_content(
        model=gemini.model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, max_output_tokens=300
        ),
    )
    return response.text


def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Requires paid credits: https://console.anthropic.com/settings/billing"""
    import anthropic

    claude = PROVIDERS["anthropic"]
    client = anthropic.Anthropic(api_key=claude.api_key)
    response = client.messages.create(
        model=claude.model,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    result = generate_answer("How long do I have to get a refund?")
    print(result)
