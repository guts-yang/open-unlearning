"""Aggregate ALTER lm-eval summaries. Missing runs stay 未复现."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _md_table import replace_measured_section

ALTER_HEADERS = ["run", "model", "wmdp_bio", "wmdp_cyber", "mmlu", "note", "status"]

ALTER_PLACEHOLDERS = [
    {
        "run": "未复现",
        "model": "zephyr-7b-beta",
        "note": "原文无完整复现配置；推定实现；论文 Cyber 24.0 / MMLU 56.4 / Bio 24.4；训练未跑",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "model": "llama3-8b",
        "note": "原文无完整复现配置；推定实现；论文 Bio 24.4 / Cyber 25.6 / MMLU 57.8；训练未跑",
        "status": "未复现",
    },
]


def infer_model(path: Path) -> str | None:
    text = str(path).lower()
    if "zephyr" in text:
        return "zephyr-7b-beta"
    if "llama3" in text or "llama-3" in text or "meta-llama-3" in text:
        return "llama3-8b"
    return None


def _metric(data: dict, *keys):
    for key in keys:
        if key in data:
            val = data[key]
            if isinstance(val, dict):
                return val.get("acc,none", val.get("acc", val.get("agg_value")))
            return val
    return None


def _empty_grid() -> list[dict]:
    rows = []
    for raw in ALTER_PLACEHOLDERS:
        row = {h: "" for h in ALTER_HEADERS}
        row.update(raw)
        rows.append(row)
    return rows


def _apply_measured(rows: list[dict], model: str, measured: dict, run: str) -> None:
    for row in rows:
        if row["model"] == model and row["status"] == "未复现":
            row["run"] = run
            for key in ("wmdp_bio", "wmdp_cyber", "mmlu"):
                if measured.get(key) not in (None, ""):
                    row[key] = measured[key]
            row["status"] = "实测（非论文配置复现）"
            if "原文无完整复现配置" not in str(row.get("note", "")):
                row["note"] = (
                    "原文无完整复现配置；" + str(row.get("note") or "")
                ).rstrip("；")
            return
    extra = {h: "" for h in ALTER_HEADERS}
    extra.update(
        {
            "run": run,
            "model": model or "",
            "status": "实测（非论文配置复现）",
            "note": "原文无完整复现配置",
            **{k: measured.get(k, "") for k in ("wmdp_bio", "wmdp_cyber", "mmlu")},
        }
    )
    rows.append(extra)


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
        default=Path("workspace/results/alter_table.md"),
    )
    args = parser.parse_args()
    rows = _empty_grid()
    if args.saves_root.exists():
        for path in sorted(args.saves_root.glob("**/LMEval_SUMMARY.json")):
            data = json.loads(path.read_text())
            measured = {
                "wmdp_bio": _metric(data, "wmdp_bio", "wmdp_bio_acc"),
                "wmdp_cyber": _metric(data, "wmdp_cyber", "wmdp_cyber_acc"),
                "mmlu": _metric(data, "mmlu", "mmlu_acc"),
            }
            _apply_measured(rows, infer_model(path) or "", measured, str(path.parent))
    replace_measured_section(args.out, ALTER_HEADERS, rows)


if __name__ == "__main__":
    main()
