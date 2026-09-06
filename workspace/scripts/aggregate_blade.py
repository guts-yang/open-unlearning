"""BLADE paper HM: hmean(MU, 1-Prob, 1-RG). Not OpenUnlearning Forget Quality."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _md_table import replace_measured_section

BLADE_HEADERS = [
    "run",
    "split",
    "seed",
    "MU",
    "Prob",
    "RG",
    "FQ",
    "HM_paper",
    "HM_paper_ref",
    "status",
]

BLADE_PLACEHOLDERS = [
    {"run": "未复现", "split": "TOFU-1B-01", "HM_paper_ref": "0.808", "status": "未复现"},
    {"run": "未复现", "split": "TOFU-1B-05", "HM_paper_ref": "0.800", "status": "未复现"},
    {"run": "未复现", "split": "TOFU-1B-10", "HM_paper_ref": "0.803", "status": "未复现"},
    {"run": "未复现", "split": "TOFU-3B-01", "HM_paper_ref": "0.846", "status": "未复现"},
    {"run": "未复现", "split": "TOFU-3B-05", "HM_paper_ref": "0.842", "status": "未复现"},
    {"run": "未复现", "split": "TOFU-3B-10", "HM_paper_ref": "0.836", "status": "未复现"},
    {"run": "未复现", "split": "MUSE-Books", "HM_paper_ref": "0.818", "status": "未复现"},
    {"run": "未复现", "split": "MUSE-News", "HM_paper_ref": "0.544", "status": "未复现"},
]

_SPLIT_PATTERNS = [
    (re.compile(r"muse[_-]news", re.I), "MUSE-News"),
    (re.compile(r"muse[_-]books", re.I), "MUSE-Books"),
    (re.compile(r"tofu[_-]?3b[_-]?10", re.I), "TOFU-3B-10"),
    (re.compile(r"tofu[_-]?3b[_-]?05", re.I), "TOFU-3B-05"),
    (re.compile(r"tofu[_-]?3b[_-]?01", re.I), "TOFU-3B-01"),
    (re.compile(r"tofu[_-]?1b[_-]?10", re.I), "TOFU-1B-10"),
    (re.compile(r"tofu[_-]?1b[_-]?05", re.I), "TOFU-1B-05"),
    (re.compile(r"tofu[_-]?1b[_-]?01", re.I), "TOFU-1B-01"),
]


def harmonic_mean(values):
    xs = [float(v) for v in values]
    if any(x <= 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def blade_tofu_hm(mu: float, prob: float, rouge: float) -> float:
    return harmonic_mean((mu, 1.0 - prob, 1.0 - rouge))


def blade_muse_hm(fk: float, vm: float, rk: float) -> float:
    return harmonic_mean((1.0 - fk, 1.0 - vm, rk))


def _nested_get(obj, *keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_tofu_metrics(eval_json: Path) -> dict:
    data = json.loads(eval_json.read_text())
    mu = _nested_get(data, "model_utility", "agg_value")
    if mu is None:
        mu = _nested_get(data, "model_utility")
        if isinstance(mu, dict):
            mu = mu.get("agg_value")
    prob = _nested_get(data, "forget_Q_A_Prob", "agg_value")
    rouge = _nested_get(data, "forget_Q_A_ROUGE", "agg_value")
    fq = _nested_get(data, "forget_quality", "agg_value")
    hm = (
        blade_tofu_hm(mu, prob, rouge)
        if None not in (mu, prob, rouge)
        else None
    )
    return {"MU": mu, "Prob": prob, "RG": rouge, "FQ": fq, "HM_paper": hm}


def load_muse_metrics(summary: Path) -> dict:
    data = json.loads(summary.read_text())
    fk = _nested_get(data, "forget_knowmem", "agg_value") or data.get("forget_knowmem")
    vm = _nested_get(data, "forget_verbmem", "agg_value") or data.get("forget_verbmem")
    rk = _nested_get(data, "retain_knowmem", "agg_value") or data.get("retain_knowmem")
    hm = (
        blade_muse_hm(fk, vm, rk)
        if None not in (fk, vm, rk) and all(isinstance(x, (int, float)) for x in (fk, vm, rk))
        else None
    )
    return {"MU": rk, "Prob": fk, "RG": vm, "FQ": "", "HM_paper": hm}


def infer_split(path: Path) -> str | None:
    text = str(path)
    for pattern, split in _SPLIT_PATTERNS:
        if pattern.search(text):
            return split
    return None


def infer_seed(path: Path) -> str:
    match = re.search(r"s(?:eed)?[_-]?(\d+)", str(path), re.I)
    return match.group(1) if match else ""


def _empty_grid() -> list[dict]:
    rows = []
    for raw in BLADE_PLACEHOLDERS:
        row = {h: "" for h in BLADE_HEADERS}
        row.update(raw)
        rows.append(row)
    return rows


def _apply_measured(rows: list[dict], split: str, measured: dict, run: str, seed: str) -> None:
    for row in rows:
        if row["split"] == split and row["status"] == "未复现":
            row["run"] = run
            row["seed"] = seed
            for key in ("MU", "Prob", "RG", "FQ", "HM_paper"):
                if measured.get(key) not in (None, ""):
                    row[key] = measured[key]
            row["status"] = "实测"
            return
    extra = {h: "" for h in BLADE_HEADERS}
    extra.update(
        {
            "run": run,
            "split": split or "",
            "seed": seed,
            "status": "实测",
            **{k: measured.get(k, "") for k in ("MU", "Prob", "RG", "FQ", "HM_paper")},
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
        default=Path("workspace/results/blade_table.md"),
    )
    args = parser.parse_args()
    rows = _empty_grid()
    if args.saves_root.exists():
        for summary in sorted(args.saves_root.glob("**/TOFU_EVAL.json")):
            metrics = load_tofu_metrics(summary)
            _apply_measured(
                rows,
                infer_split(summary) or "",
                metrics,
                str(summary.parent),
                infer_seed(summary),
            )
        for summary in sorted(args.saves_root.glob("**/MUSE_SUMMARY.json")):
            metrics = load_muse_metrics(summary)
            _apply_measured(
                rows,
                infer_split(summary) or "",
                metrics,
                str(summary.parent),
                infer_seed(summary),
            )
    replace_measured_section(args.out, BLADE_HEADERS, rows)


if __name__ == "__main__":
    main()
