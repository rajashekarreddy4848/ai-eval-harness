"""The Hugging Face Space must run the same pipeline the eval suites test. No API calls."""

from pathlib import Path

import pytest

from scripts.sync_space import SPACE_FILES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", SPACE_FILES)
def test_space_copy_matches_app(name):
    app_file = ROOT / "app" / name
    space_file = ROOT / "huggingface_space" / "app" / name
    assert space_file.exists(), f"huggingface_space/app/{name} is missing; run python scripts/sync_space.py"
    assert space_file.read_bytes() == app_file.read_bytes(), (
        f"huggingface_space/app/{name} differs from app/{name}; run python scripts/sync_space.py"
    )
