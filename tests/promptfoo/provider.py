"""Custom promptfoo provider: bridges promptfoo's test runner to our RAG pipeline.

promptfoo calls `call_api(prompt, options, context)` for every test case and
expects a dict with an "output" key. See:
https://www.promptfoo.dev/docs/providers/python/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag_pipeline import generate_answer  # noqa: E402


def call_api(prompt, options, context):
    # Red-team cases may carry a poisoned knowledge-base doc (indirect prompt injection).
    inject_doc = ((context or {}).get("vars") or {}).get("inject_doc")
    result = generate_answer(prompt, injected_docs=[inject_doc] if inject_doc else None)
    return {"output": result["answer"]}
