"""Pytest configuration for the open-unlearning test suite.

The package sources live under ``src/`` and ``setup.py`` uses a plain
``find_packages()`` (no ``package_dir`` mapping), so running ``pytest tests/``
from the repo root cannot resolve ``evals``, ``model``, ``trainer``, ... unless
``src/`` is added to ``sys.path``. This module does exactly that.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
