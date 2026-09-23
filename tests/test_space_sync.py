"""The Hugging Face Space must run the same pipeline the eval suites test. No API calls."""

import re
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


def test_space_ships_every_app_module_it_imports():
    """Comparing listed files can't notice a new import of a file that isn't listed."""
    sources = [ROOT / "huggingface_space" / "app.py"] + [ROOT / "app" / n for n in SPACE_FILES]
    imported = {
        f"{module}.py"
        for path in sources
        for module in re.findall(r"^\s*(?:from|import) app\.(\w+)", path.read_text(), re.MULTILINE)
    }
    assert imported <= set(SPACE_FILES), f"add to SPACE_FILES: {sorted(imported - set(SPACE_FILES))}"
