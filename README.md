# AI Eval Harness — Automated Quality Testing for LLM/RAG Applications

A small RAG-based FAQ chatbot wrapped in a full **automated test & evaluation
pipeline**, built to demonstrate applying test automation / SDET principles
to LLM-based systems.

This project answers the interview question *"have you tested AI systems?"*
with a working repo instead of a claim.

## Why this exists

Traditional QA automation checks deterministic outputs (`assertEquals`,
status codes, DOM elements). LLM outputs are non-deterministic, so testing
them needs a different toolkit — this repo covers the three main approaches:

| Layer | Tool | What it checks | Analogous to (SDET terms) |
|---|---|---|---|
| Regression assertions | [promptfoo](https://promptfoo.dev) | Exact/contains checks, LLM-graded rubrics, latency | Assertion-based test cases in a CI gate |
| Output quality metrics | [deepeval](https://deepeval.com) | Hallucination, answer relevancy, faithfulness | pytest suite with pass/fail thresholds |
| RAG-specific scoring | [ragas](https://docs.ragas.io) | Context precision/recall, faithfulness (dataset-level) | A quality dashboard / test report across a golden set |

## Architecture

```
app/
  knowledge_base.py   # 6 sample FAQ docs (refunds, shipping, accounts, etc.)
  rag_pipeline.py      # TF-IDF retrieval + Claude generation (kept simple on purpose)
tests/
  test_deepeval.py     # pytest suite: hallucination / relevancy / faithfulness gates
  test_ragas.py         # scored report across a golden eval set -> ragas_report.csv
  promptfoo/
    promptfooconfig.yaml  # regression test cases (rubrics, contains, latency)
    provider.py            # bridges promptfoo -> our RAG pipeline
.github/workflows/eval.yml # CI: runs all three suites on every push/PR
```

The RAG app itself is intentionally minimal (TF-IDF instead of a vector DB,
6 FAQ docs instead of a real corpus) — the point of this project is the
**test harness around it**, not the app.

## Setup

```bash
pip install -r requirements.txt
npm install -g promptfoo   # only needed for the promptfoo suite

export ANTHROPIC_API_KEY=your_key_here
```

## Running the suites

```bash
# 1. Regression tests (fast, deterministic assertions + rubric grading)
cd tests/promptfoo && promptfoo eval

# 2. Quality gates as pytest (hallucination / relevancy / faithfulness thresholds)
pytest tests/test_deepeval.py -v

# 3. Scored quality report across the golden eval set
python tests/test_ragas.py
```

CI runs all three automatically on every push via
[`.github/workflows/eval.yml`](.github/workflows/eval.yml) and uploads the
ragas report as a build artifact.

## What this demonstrates (for interviews / resume)

- Designing a **golden test set** for a non-deterministic system
- Choosing the right metric for the right failure mode (hallucination vs.
  relevancy vs. retrieval quality — these catch different bugs)
- Wiring LLM evals into **CI/CD** the same way you'd wire up any automated
  test suite, with pass/fail gates and artifact reporting
- Understanding RAG failure modes: bad retrieval (context precision/recall)
  vs. bad generation (faithfulness/hallucination) — and testing each
  independently so a failure tells you *where* the bug is, not just *that*
  there is one

## Resume bullet (starting point)

> Built an automated evaluation/testing harness for a RAG-based LLM
> application using promptfoo, deepeval, and ragas, covering regression
> testing, hallucination/faithfulness detection, and retrieval quality
> scoring, wired into a CI pipeline with automated quality gates.

## Possible extensions

- Swap TF-IDF retrieval for a real vector DB (Chroma/Pinecone) + embeddings
- Add an adversarial/red-team test set (prompt injection, jailbreak attempts)
- Track eval scores over time (regression detection across commits)
- Add a second LLM provider to compare quality/cost tradeoffs
