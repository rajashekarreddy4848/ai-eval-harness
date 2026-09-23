"""Gradio chat UI for the FAQ RAG bot, deployed as a Hugging Face Space.

This is the same RAG pipeline used in the eval harness (see the main repo's
README for the promptfoo/deepeval/ragas test suite) -- this file just wraps
it in a live, clickable demo.
"""

import gradio as gr

from app.rag_pipeline import generate_answer

EXAMPLES = [
    "How many days do I have to request a refund?",
    "How much does the Pro plan cost?",
    "How do I reset my password?",
    "What happens after I delete my account?",
]


def chat(message, history):
    result = generate_answer(message)
    return result["answer"]


demo = gr.ChatInterface(
    fn=chat,
    title="FAQ Support Bot (RAG demo)",
    description=(
        "A small retrieval-augmented chatbot over a 6-document FAQ knowledge base "
        "(refunds, shipping, accounts, passwords, plans, data export). "
        "Built alongside a full automated eval/testing harness (promptfoo, deepeval, ragas) "
        "-- see the [GitHub repo](https://github.com/rajashekarreddy4848/ai-eval-harness) for the test suite and CI pipeline."
    ),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
