"""RAG-specific quality report using ragas.

Unlike the pytest suites (pass/fail gates), this script produces a scored
report across the whole eval set -- the kind of artifact you'd screenshot
for a resume/portfolio or paste into a PR description.

Run with: python tests/test_ragas.py

Metrics: faithfulness (generation) and context precision/recall (retrieval).
Answer relevancy is left to the deepeval suite: ragas computes it with an
embeddings model, which defaults to OpenAI and would need a separate key.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    context_precision,
    context_recall,
    faithfulness,
)

from app.eval_model import get_ragas_llm  # noqa: E402
from app.providers import generator_provider, judge_provider  # noqa: E402
from app.rag_pipeline import generate_answer  # noqa: E402

METRICS = ["faithfulness", "context_precision", "context_recall"]

EVAL_SET = [
    {
        "question": "How many days do I have to request a refund?",
        "ground_truth": "You have 30 days from purchase to request a full refund.",
    },
    {
        "question": "How long does standard shipping take?",
        "ground_truth": "Standard shipping takes 5-7 business days within the US.",
    },
    {
        "question": "How do I delete my account?",
        "ground_truth": "Go to Settings > Privacy > Delete Account. There is a 14-day grace period before it's permanent.",
    },
    {
        "question": "How much is the Pro plan?",
        "ground_truth": "The Pro plan is $12/month for a single user with full features.",
    },
]


def build_dataset() -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in EVAL_SET:
        result = generate_answer(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def main():
    generator, judge = generator_provider(), judge_provider()
    print(f"Generator: {generator.name}:{generator.model} | Judge: {judge.name}:{judge.model}")

    dataset = build_dataset()
    report = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=get_ragas_llm(),
    )
    df = report.to_pandas()
    print(df[["user_input", *METRICS]])

    out_path = ROOT / "ragas_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full report to {out_path}")

    avg_scores = df[METRICS].mean()
    print("\nAverage scores:")
    print(avg_scores)


if __name__ == "__main__":
    main()
