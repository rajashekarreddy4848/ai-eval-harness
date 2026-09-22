"""RAG-specific quality report using ragas.

Unlike the pytest suites (pass/fail gates), this script produces a scored
report across the whole eval set -- the kind of artifact you'd screenshot
for a resume/portfolio or paste into a PR description.

Run with: python tests/test_ragas.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.rag_pipeline import generate_answer

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
    dataset = build_dataset()
    report = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = report.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

    out_path = Path(__file__).resolve().parents[1] / "ragas_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full report to {out_path}")

    avg_scores = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
    print("\nAverage scores:")
    print(avg_scores)


if __name__ == "__main__":
    main()
