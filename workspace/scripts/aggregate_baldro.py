"""Aggregate BalDRO TOFU FQ/MU from EVAL JSON. Empty cells stay 未复现."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _md_table import replace_measured_section

BALDRO_HEADERS = [
    "run",
    "split",
    "method",
    "FQ",
    "MU",
    "FQ_paper",
    "MU_paper",
    "status",
]

BALDRO_PLACEHOLDERS = [
    {
        "run": "未复现",
        "split": "TOFU-01",
        "method": "NPO",
        "FQ_paper": "0.7659",
        "MU_paper": "0.5775",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-01",
        "method": "NPO+G",
        "FQ_paper": "0.9188",
        "MU_paper": "0.6126",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-01",
        "method": "NPO+DV",
        "FQ_paper": "0.9900",
        "MU_paper": "0.5815",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-05",
        "method": "NPO",
        "FQ_paper": "0.6284",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-05",
        "method": "NPO+DV",
        "FQ_paper": "0.9646",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-01",
        "method": "SimNPO+DV",
        "FQ_paper": "0.5786",
        "MU_paper": "0.5917",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "split": "TOFU-01",
        "method": "SimNPO+G",
        "FQ_paper": "0.5786",
        "MU_paper": "0.5651",
        "status": "未复现",
    },
]


def _agg(data, key):
    val = data.get(key)
    if isinstance(val, dict):
        return val.get("agg_value")
    return val


def infer_split(path: Path) -> str | None:
    text = str(path).lower()
    if re.search(r"forget05|tofu0?5", text):
        return "TOFU-05"
    if re.search(r"forget10|tofu10", text):
        return "TOFU-10"
    if re.search(r"forget01|tofu0?1", text):
        return "TOFU-01"
    return None


def infer_method(path: Path) -> str | None:
    text = str(path).lower()
    if "simnpo_dv" in text or "drsimnpo" in text:
        return "SimNPO+DV"
    if "simnpo_g" in text or "groupsimnpo" in text:
        return "SimNPO+G"
    if "npo_dv" in text or "npo+dv" in text or "drnpo" in text:
        return "NPO+DV"
    if "npo_g" in text or "npo+g" in text or "groupnpo" in text:
        return "NPO+G"
    if re.search(r"(^|[_-])npo([_-]|$)", text):
        return "NPO"
    return None


def _empty_grid() -> list[dict]:
    rows = []
    for raw in BALDRO_PLACEHOLDERS:
        row = {h: "" for h in BALDRO_HEADERS}
        row.update(raw)
        rows.append(row)
    return rows


def _apply_measured(rows: list[dict], split: str, method: str, measured: dict, run: str) -> None:
    for row in rows:
        if row["split"] == split and row["method"] == method and row["status"] == "未复现":
            row["run"] = run
            if measured.get("FQ") not in (None, ""):
                row["FQ"] = measured["FQ"]
            if measured.get("MU") not in (None, ""):
                row["MU"] = measured["MU"]
            row["status"] = "实测"
            return
    extra = {h: "" for h in BALDRO_HEADERS}
    extra.update(
        {
            "run": run,
            "split": split or "",
            "method": method or "",
            "FQ": measured.get("FQ", ""),
            "MU": measured.get("MU", ""),
            "status": "实测",
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
        default=Path("workspace/results/baldro_table.md"),
    )
    args = parser.parse_args()
    rows = _empty_grid()
    if args.saves_root.exists():
        for path in sorted(args.saves_root.glob("baldro_*/**/TOFU_EVAL.json")):
            data = json.loads(path.read_text())
            method = infer_method(path)
            if not method:
                continue
            _apply_measured(
                rows,
                infer_split(path) or "",
                method,
                {"FQ": _agg(data, "forget_quality"), "MU": _agg(data, "model_utility")},
                str(path.parent),
            )
    replace_measured_section(args.out, BALDRO_HEADERS, rows)


if __name__ == "__main__":
    main()
