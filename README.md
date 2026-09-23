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
| RAG-specific scoring | [ragas](https://docs.ragas.io) | Context precision/recall, faithfulness (dataset-level). Answer relevancy is left to deepeval because ragas needs an embeddings model for it. | A quality dashboard / test report across a golden set |

## Architecture

```
app/
  knowledge_base.py   # 6 sample FAQ docs (refunds, shipping, accounts, etc.)
  rag_pipeline.py      # retrieval + LLM generation (kept simple on purpose)
  retrieval.py         # lexical retrievers: word, word_stop, char n-gram, hybrid (default)
  providers.py         # picks the generator and a *different* judge model
  eval_model.py        # points deepeval and ragas at the judge (not OpenAI)
evals/
  golden_set.json      # 35 labeled questions shared by every suite
  golden.py            # loader, smoke/full subsets, output normalization
  holdout_set.json     # 15 questions written before retrieval tuning, scored once
  compare_retrievers.py # hit@k and MRR for every retriever (no API calls)
tests/
  test_retrieval.py    # does retrieval return the right doc? (no LLM, free)
  test_golden_set.py   # checks the labels themselves (unique, docs exist, facts in docs)
  test_deepeval.py     # pytest suite: relevancy / faithfulness / hallucination / correctness gates
  test_ragas.py         # scored report across the golden set -> ragas_report.csv
  test_providers.py     # unit tests for generator/judge selection (no API calls)
  test_space_sync.py    # fails if the Hugging Face Space copy drifts from app/
  test_judge_retry.py   # unit tests for the judge's rate-limit backoff
  promptfoo/
    promptfooconfig.yaml  # regression test cases (rubrics, contains, latency)
    provider.py            # bridges promptfoo -> our RAG pipeline
    golden_tests.py        # turns the golden set into promptfoo test cases
scripts/sync_space.py     # copies app/ into huggingface_space/app/
huggingface_space/        # Gradio demo deployed as a Hugging Face Space
.github/workflows/eval.yml # CI: unit job on every PR, then the three eval suites
```

### Generator vs. judge

LLM-graded metrics need a second model to do the grading. If the model that
wrote an answer also grades it, scores come out too kind (self-grading bias).
`app/providers.py` picks the generator from the first available key (Groq,
then Gemini, then Anthropic) and the judge from the next one. With only one key
the suites still run but warn that the model is grading itself. CI uses Groq
(`openai/gpt-oss-120b`) to answer and Gemini (`gemini-3.5-flash-lite`) to judge.
`JUDGE_MODEL` picks a different judge model, including one on the same provider.

Free tiers allow only a handful of judge calls per minute, so the judge waits
and retries on rate-limit errors instead of reporting them as failed tests. The
deepeval suite takes about 4 minutes on free tiers for that reason.

### Golden set

`evals/golden_set.json` holds 35 labeled questions, and every suite reads it:

| Category | Cases | What it probes |
|---|---|---|
| single-doc | 20 | One fact from one doc, every doc covered |
| paraphrase | 2 | Casual wording and typos ("how fast is expres shiping") |
| multi-doc | 3 | Answers that need two docs |
| out-of-scope | 5 | Questions the FAQ doesn't cover; the bot must say it doesn't know |
| false-premise | 3 | "Why is the refund window 60 days?" must be corrected, not accepted |
| numeric-reasoning | 2 | "Team plan for 5 users" needs $9 x 5 = $45 |

Each case records the expected behavior (answer or decline), the docs that
should be retrieved, facts a correct answer must contain, and a reference
answer. `tests/test_golden_set.py` checks the labels themselves, e.g. that every
required fact actually appears in the expected doc.

LLM-graded suites are slow on free tiers, so deepeval and ragas run 8 "smoke"
cases by default; `EVAL_SUBSET=full` runs everything. Retrieval tests and
promptfoo always run the full set.

### Findings from the first golden-set run

- **Every end-to-end failure was a retrieval miss.** Retrieval found the right
  doc for 23 of 30 answerable questions. Promptfoo failed exactly those 7 and
  nothing else, so no failure came from the model misusing a correct doc.
- **Root cause of 4 of the 7:** TF-IDF has no stemming. Questions say "plan",
  the pricing doc only says "plans", and the Data Export doc says "Free plan",
  so it wins. The other 3 are a synonym ("money back" vs "refund"), typos, and a
  long question whose filler words pull in other docs.
- **The model failed safely.** With the wrong docs it said "I don't know" instead
  of inventing a price.
- **Relevancy and faithfulness can't see a missed answer.** deepeval scored
  "I don't know." to "How much does the Pro plan cost?" 1.0 on both metrics,
  because neither metric knows the right answer. A GEval correctness check
  against the reference answer scored it 0.0, so it's now part of the suite.
- **Tools disagree on refusals.** For that same answer ragas faithfulness was
  0.0 and deepeval faithfulness was 1.0.
- **String checks need normalized text.** The model writes "30<U+202F>days" and
  "I don<U+2019>t know", which made correct answers fail `icontains` and let a
  wrong answer pass `not-icontains`. Both promptfoo and the Python checks now
  normalize output first.

The 7 retrieval misses were marked `xfail(strict=True)` in `tests/test_retrieval.py`
with their causes, so fixing one makes the suite demand the marker's removal.

### Fixing retrieval, measured

Four lexical retrievers were compared with `evals/compare_retrievers.py`. The
15-question held-out set was written **before** any tuning and scored once,
after the choice was made:

| Retriever | Golden hit@2 (tuned on) | Held-out hit@2 |
|---|---|---|
| word (original) | 23/30 (77%) | 14/15 |
| word_stop (drop stop words) | 27/30 (90%) | 13/15 |
| char (3-5 char n-grams) | 26/30 (87%) | 15/15 |
| **hybrid** (word_stop + char, averaged) | **28/30 (93%)** | **15/15** |

`word_stop` is the cautionary one: it improved on the golden set it was picked
on and got *worse* than the original on held-out questions. Hybrid is best on
both, so it's the default. Held-out gains are small (14 to 15) because the
original already did well on those questions; they are easier than the golden
set's typo and multi-doc cases.

While comparing, the harness itself had a bug: when a query shares no words with
any doc, every score is 0 and the sort returns the first docs in the knowledge
base. `word_stop` got two "hits" that way. Docs with score 0 are no longer
retrieved, so those queries now get no context and an honest "I don't know".

End to end, same golden set, before and after:

| Measure | word | hybrid |
|---|---|---|
| promptfoo, all 36 cases | 29 pass | 33 pass |
| deepeval smoke (4 metrics) | 27/29 | 29/29 |
| ragas faithfulness / context recall | 0.86 / 0.81 | 1.00 / 1.00 |
| ragas context precision | 0.64 | 0.79 |

Still open:

- **REF-04** ("money back") and **FPR-03** ("Enterprise plan") need meaning, not
  spelling; lexical retrieval can't get them. Embeddings would be the next step.
- **Ranking:** for plan questions hybrid returns the pricing doc second, behind
  Data Export. Answers are right, but that's why context precision is below 1.
- **OOS-05, a flaky failure that one run hid.** "Write me a short poem about
  shipping" passed in the first golden run, then failed. Rerun 6 times, the bot
  wrote the poem 6/6 with the old retriever and 4/6 with hybrid. It's a prompt
  problem (a support bot shouldn't do creative writing), not a retrieval
  regression, and single runs of LLM evals can't be trusted for it.

The RAG app itself is intentionally minimal (lexical TF-IDF instead of a vector DB,
6 FAQ docs instead of a real corpus) — the point of this project is the
**test harness around it**, not the app.

## Setup

```bash
pip install -r requirements.txt
npm install -g promptfoo   # only needed for the promptfoo suite

cp .env.example .env       # then add GROQ_API_KEY and GEMINI_API_KEY (both free)
```

## Running the suites

```bash
# 0. No API keys needed: golden-set checks, retrieval tests, unit tests
pytest tests/test_golden_set.py tests/test_retrieval.py tests/test_providers.py tests/test_space_sync.py tests/test_judge_retry.py

# 1. Regression tests (fast, deterministic assertions + rubric grading)
#    promptfoo's Gemini grader reads GOOGLE_API_KEY: export GOOGLE_API_KEY=$GEMINI_API_KEY
cd tests/promptfoo && promptfoo eval

# 2. Quality gates as pytest (relevancy / faithfulness / hallucination / correctness)
pytest tests/test_deepeval.py -v                      # smoke subset, ~5 min on free tiers
EVAL_SUBSET=full pytest tests/test_deepeval.py -v     # all 35 cases

# 3. Scored quality report across the golden eval set
python tests/test_ragas.py
```

CI runs all three automatically on every push via
[`.github/workflows/eval.yml`](.github/workflows/eval.yml) and uploads the
ragas report as a build artifact. It needs `GROQ_API_KEY` and `GEMINI_API_KEY`
as repository secrets.

After changing anything in `app/`, run `python scripts/sync_space.py` so the
Hugging Face Space runs the same code the tests check.

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

- Add embeddings (hybrid with the lexical retriever) for the synonym cases REF-04 and FPR-03
- Add an adversarial/red-team test set (prompt injection, jailbreak attempts)
- Run each LLM-graded case several times and gate on pass rate, to catch flaky failures like OOS-05
- Track eval scores over time (regression detection across commits)
