#!/usr/bin/env python3
"""Print ``hits total`` for official runs of one method. Used by ou_p1_table3.sh."""
import json
import sys
from pathlib import Path

TARGETS = {
    "SimNPO": {"Agg": 0.53, "Mem": 0.32, "Priv": 0.63, "Utility": 1.00},
    "RMU": {"Agg": 0.52, "Mem": 0.47, "Priv": 0.50, "Utility": 0.61},
}

method = sys.argv[1] if len(sys.argv) > 1 else "SimNPO"
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
tgt = TARGETS[method]
hits = n = 0
p = Path("results/ou_table3_runs.jsonl")
if p.exists():
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source") != "official" or r.get("method") != method:
            continue
        n += 1
        if all(abs(r[d] - tgt[d]) <= tol for d in tgt):
            hits += 1
print(hits, n)
