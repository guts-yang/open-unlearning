from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "workspace" / "src")]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO
