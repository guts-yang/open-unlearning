#!/usr/bin/env python3
"""Exit 0 iff local dir has a complete HF causal-LM checkpoint (no .incomplete)."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
if not (d / "config.json").is_file():
    sys.exit(1)
if any(d.rglob("*.incomplete")):
    sys.exit(1)
for name in ("model.safetensors", "pytorch_model.bin"):
    p = d / name
    if p.is_file() and p.stat().st_size > 1_000_000:
        sys.exit(0)
idx = d / "model.safetensors.index.json"
if idx.is_file():
    files = set(json.loads(idx.read_text()).get("weight_map", {}).values())
    if files and all((d / f).is_file() and (d / f).stat().st_size > 0 for f in files):
        sys.exit(0)
sys.exit(1)
