"""BLADE paper HM: hmean(MU, 1-Prob, 1-RG). Not OpenUnlearning Forget Quality."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    return {
        "MU": mu,
        "Prob": prob,
        "RG": rouge,
        "FQ": fq,
        "HM": blade_tofu_hm(mu, prob, rouge)
        if None not in (mu, prob, rouge)
        else None,
    }


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
        default=Path("workspace/results/blade_table.csv"),
    )
    args = parser.parse_args()
    rows = []
    if args.saves_root.exists():
        for summary in sorted(args.saves_root.glob("**/TOFU_EVAL.json")):
            metrics = load_tofu_metrics(summary)
            metrics["run"] = str(summary.parent)
            metrics["file"] = str(summary)
            rows.append(metrics)
        for summary in sorted(args.saves_root.glob("**/MUSE_SUMMARY.json")):
            data = json.loads(summary.read_text())
            row = {"run": str(summary.parent), "file": str(summary)}
            row.update({k: data.get(k) for k in data})
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "file", "MU", "Prob", "RG", "FQ", "HM"]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if not rows:
            writer.writerow(
                {
                    "run": "未复现",
                    "file": "",
                    "MU": "",
                    "Prob": "",
                    "RG": "",
                    "FQ": "",
                    "HM": "",
                }
            )
        else:
            writer.writerows(rows)


if __name__ == "__main__":
    main()
