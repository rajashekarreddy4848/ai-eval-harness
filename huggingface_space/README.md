---
title: FAQ RAG Eval Bot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# FAQ Support Bot (RAG demo)

A small retrieval-augmented chatbot (TF-IDF retrieval + LLM generation) over
a 6-document FAQ knowledge base. Built as the live demo half of a project
whose real focus is the automated eval/testing harness around it
(promptfoo, deepeval, ragas) — see the main GitHub repo for the test suite,
CI pipeline, and quality reports.

Try asking about refunds, shipping times, account deletion, password resets,
subscription plans, or data export.

It runs the same code the tests check: the hybrid retriever and the hardened
system prompt (v2) that blocked every poisoned-document injection in the red-team
suite. Test suite and results: https://github.com/rajashekarreddy4848/ai-eval-harness

Needs a `GROQ_API_KEY` secret in the Space settings.
