"""Copy the app modules the Hugging Face Space needs into huggingface_space/app/.

The Space is deployed from its own folder, so it carries a copy of the RAG
pipeline. Run this after changing app/, and tests/test_space_sync.py fails in
CI if the copy drifts.

Run with: python scripts/sync_space.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ["__init__.py", "knowledge_base.py", "providers.py", "rag_pipeline.py", "retrieval.py"]


def main():
    target = ROOT / "huggingface_space" / "app"
    target.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / "app" / name, target / name)
        print(f"copied app/{name}")


if __name__ == "__main__":
    main()
