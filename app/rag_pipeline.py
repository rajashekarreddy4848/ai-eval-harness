"""Minimal RAG pipeline: TF-IDF retrieval + Claude generation.

Kept deliberately simple (no vector DB, no embeddings API) so the project
runs cheaply and the focus stays on the EVAL/TEST harness, not the app itself.
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
    """Calls Anthropic's Claude API. Requires ANTHROPIC_API_KEY in the environment."""
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
