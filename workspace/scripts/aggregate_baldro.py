"""Aggregate BalDRO TOFU FQ/MU from EVAL JSON. Empty cells stay 未复现."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _agg(data, key):
    val = data.get(key)
    if isinstance(val, dict):
        return val.get("agg_value")
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saves-root",
        type=Path,
        default=Path("workspace/saves/unlearn"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/results/baldro_table.csv"),
    )
    args = parser.parse_args()
    rows = []
    if args.saves_root.exists():
        for path in sorted(args.saves_root.glob("**/TOFU_EVAL.json")):
            data = json.loads(path.read_text())
            rows.append(
                {
                    "run": str(path.parent),
                    "FQ": _agg(data, "forget_quality"),
                    "MU": _agg(data, "model_utility"),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "FQ", "MU"])
        writer.writeheader()
        if not rows:
            writer.writerow({"run": "未复现", "FQ": "", "MU": ""})
        else:
            writer.writerows(rows)


if __name__ == "__main__":
    main()
