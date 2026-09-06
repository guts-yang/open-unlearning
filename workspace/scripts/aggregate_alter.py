"""Aggregate ALTER lm-eval summaries. Missing runs stay 未复现."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
        default=Path("workspace/results/alter_table.csv"),
    )
    args = parser.parse_args()
    rows = []
    if args.saves_root.exists():
        for path in sorted(args.saves_root.glob("**/LMEval_SUMMARY.json")):
            data = json.loads(path.read_text())
            rows.append({"run": str(path.parent), **{str(k): v for k, v in data.items()}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "wmdp_bio", "wmdp_cyber", "mmlu", "note"]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if not rows:
            writer.writerow(
                {
                    "run": "未复现",
                    "wmdp_bio": "",
                    "wmdp_cyber": "",
                    "mmlu": "",
                    "note": "no LMEval_SUMMARY.json yet",
                }
            )
        else:
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
