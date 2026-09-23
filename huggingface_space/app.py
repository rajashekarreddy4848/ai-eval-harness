"""Gradio chat UI for the FAQ RAG bot, deployed on Render (see render.yaml).

This is the same RAG pipeline used in the eval harness (see the main repo's
README for the promptfoo/deepeval/ragas test suite) -- this file just wraps
it in a live, clickable demo.
"""

import os

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
    # Hosts like Render pass the port in PORT and need the app reachable from outside
    # the container. Locally, with no PORT, stay on localhost:7860.
    port = os.environ.get("PORT")
    demo.launch(server_name="0.0.0.0" if port else "127.0.0.1", server_port=int(port or 7860))
