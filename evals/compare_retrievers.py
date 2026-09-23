"""Compare retrievers on the golden set (and, once, on the held-out set). No API calls.

Run with: python evals/compare_retrievers.py
          python evals/compare_retrievers.py --holdout   # only after choosing a retriever

hit@k  share of questions whose expected docs are ALL in the top k
MRR    mean reciprocal rank of the first expected doc (1.0 = always ranked first)

The golden set is what retrievers were tuned on, so its numbers are optimistic.
evals/holdout_set.json was written before tuning and is the honest estimate.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import RETRIEVER_NAMES, build_retriever, top_k  # noqa: E402
from evals.golden import answer_cases  # noqa: E402

HOLDOUT_PATH = ROOT / "evals" / "holdout_set.json"


def evaluate(name: str, cases: list[dict], k: int) -> dict:
    retriever = build_retriever(name)
    hits, rr, misses = 0, 0.0, []
    for case in cases:
        ranked = [d["title"] for d in top_k(retriever, case["question"], len(retriever.docs))]
        if all(t in ranked[:k] for t in case["expected_docs"]):
            hits += 1
        else:
            misses.append(case["id"])
        ranks = [ranked.index(t) for t in case["expected_docs"] if t in ranked]
        rr += 1 / (min(ranks) + 1) if ranks else 0.0
    return {"hit": hits / len(cases), "hits": hits, "mrr": rr / len(cases), "misses": misses}


def table(title: str, cases: list[dict], ks=(2, 3)) -> None:
    print(f"\n{title} ({len(cases)} questions)")
    print(f"{'retriever':<10} " + " ".join(f"{'hit@' + str(k):>9}" for k in ks) + f" {'MRR':>6}  misses at k=2")
    for name in RETRIEVER_NAMES:
        results = {k: evaluate(name, cases, k) for k in ks}
        cells = " ".join(f"{results[k]['hits']:>3}/{len(cases):<2} {results[k]['hit']:>3.0%}" for k in ks)
        print(f"{name:<10} {cells} {results[2]['mrr']:>6.2f}  {', '.join(results[2]['misses']) or '-'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true", help="also score the held-out set")
    args = parser.parse_args()
    table("Golden set (tuning data)", answer_cases("full"))
    if args.holdout:
        table("Held-out set", json.loads(HOLDOUT_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
