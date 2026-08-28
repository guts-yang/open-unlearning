"""Append-only audit ledger for MOGP-U trial provenance."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class Ledger:
    def __init__(self, root: str | Path, tier: str):
        if tier not in {"search", "validation", "sealed_test"}:
            raise ValueError("Invalid MOGP-U ledger tier")
        self.path = Path(root) / tier / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: dict[str, Any]) -> None:
        required = {"candidate_hash", "generation", "tier", "output_dir", "fq_feasible"}
        missing = required - set(entry)
        if missing:
            raise ValueError(f"Ledger entry missing: {sorted(missing)}")
        if entry["tier"] != self.path.parent.name:
            raise ValueError("Ledger tier does not match ledger location")
        line = json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
        previous = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            handle.write(previous + line)
            temporary = Path(handle.name)
        temporary.replace(self.path)
