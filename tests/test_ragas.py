"""RAG-specific quality report using ragas.

Unlike the pytest suites (pass/fail gates), this script produces a scored
report across the whole eval set -- the kind of artifact you'd screenshot
for a resume/portfolio or paste into a PR description.

Run with: python tests/test_ragas.py               (smoke subset of the golden set)
          EVAL_SUBSET=full python tests/test_ragas.py

Cases are the answerable questions in evals/golden_set.json. Metrics: faithfulness (generation) and context precision/recall (retrieval).
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
from ragas import RunConfig, evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    context_precision,
    context_recall,
    faithfulness,
)

from app.eval_model import get_ragas_llm  # noqa: E402
from app.providers import generator_provider, judge_provider  # noqa: E402
from app.rag_pipeline import generate_answer  # noqa: E402
from evals.golden import answer_cases  # noqa: E402

METRICS = ["faithfulness", "context_precision", "context_recall"]


def build_dataset(cases: list[dict]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in cases:
        result = generate_answer(item["question"])
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def main():
    generator, judge = generator_provider(), judge_provider()
    print(f"Generator: {generator.name}:{generator.model} | Judge: {judge.name}:{judge.model}")

    cases = answer_cases()
    dataset = build_dataset(cases)
    report = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=get_ragas_llm(),
        # Few parallel judge calls: free tiers rate-limit bursts.
        run_config=RunConfig(max_workers=2),
    )
    df = report.to_pandas()
    df.insert(0, "id", [c["id"] for c in cases])
    df.insert(1, "category", [c["category"] for c in cases])
    print(df[["id", *METRICS]].to_string(index=False))
    print("\nAverage by category:")
    print(df.groupby("category")[METRICS].mean().round(2).to_string())

    out_path = ROOT / "ragas_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full report to {out_path}")

    avg_scores = df[METRICS].mean()
    print("\nAverage scores:")
    print(avg_scores)


if __name__ == "__main__":
    main()
