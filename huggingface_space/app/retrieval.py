"""Retrievers for the FAQ bot, all lexical (no embeddings API, no model download).

  word       word TF-IDF, the original retriever. No stemming, so "plan" does not
             match "plans", and typos match nothing.
  word_stop  word TF-IDF without English stop words, so filler words in long
             questions stop pulling in unrelated docs.
  char       TF-IDF over character 3-5-grams inside words: "plan"/"plans" and
             "shiping"/"shipping" share most of their n-grams.
  hybrid     average of word_stop and char similarity scores.

evals/compare_retrievers.py measures each one on the golden set; RETRIEVER
picks one at runtime (default below).
"""

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.knowledge_base import DOCS

DEFAULT_RETRIEVER = "hybrid"


class TfidfRetriever:
    def __init__(self, docs: list[dict], **vectorizer_options):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(**vectorizer_options)
        self.matrix = self.vectorizer.fit_transform(d["text"] for d in docs)

    def scores(self, query: str) -> np.ndarray:
        return cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()


class HybridRetriever:
    def __init__(self, docs: list[dict], parts: list):
        self.docs = docs
        self.parts = parts

    def scores(self, query: str) -> np.ndarray:
        return np.mean([p.scores(query) for p in self.parts], axis=0)


def build_retriever(name: str, docs: list[dict] = DOCS):
    word = dict()
    word_stop = dict(stop_words="english")
    char = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    if name == "word":
        return TfidfRetriever(docs, **word)
    if name == "word_stop":
        return TfidfRetriever(docs, **word_stop)
    if name == "char":
        return TfidfRetriever(docs, **char)
    if name == "hybrid":
        return HybridRetriever(docs, [TfidfRetriever(docs, **word_stop), TfidfRetriever(docs, **char)])
    raise ValueError(f"Unknown retriever {name!r}; use one of {', '.join(RETRIEVER_NAMES)}")


RETRIEVER_NAMES = ("word", "word_stop", "char", "hybrid")


def top_k(retriever, query: str, k: int) -> list[dict]:
    """Up to k docs, best first. Docs that share nothing with the query (score 0) are
    never returned: otherwise a query matching nothing gets the first docs in the
    knowledge base, and an eval counts that luck as a hit."""
    scores = retriever.scores(query)
    order = np.argsort(-scores, kind="stable")[:k]
    return [retriever.docs[i] for i in order if scores[i] > 0]


def active_retriever_name() -> str:
    return os.environ.get("RETRIEVER") or DEFAULT_RETRIEVER
