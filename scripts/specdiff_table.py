#!/usr/bin/env python3
"""Summarize the three TOFU-1B SpecDiff seeds."""

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAVES = Path(os.environ.get("SAVES", "/root/autodl-tmp/saves"))
RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def main():
    rows = []
    missing = []
    for seed in SEEDS:
        path = (
            SAVES
            / "unlearn"
            / f"tofu_1B_SpecDiff_forget10_seed{seed}"
            / "evals"
            / "ou_aggregate.json"
        )
        if not path.is_file():
            missing.append(seed)
            continue
        result = json.loads(path.read_text())
        specgap = result["SpecGap"]
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(path.parents[1]),
                "SpecGap_f": specgap["forget"]["mean"],
                "SpecGap_f_ci95": specgap["forget"]["ci95"],
                "SpecGap_r": specgap["retain"]["mean"],
                "SpecGap_r_ci95": specgap["retain"]["ci95"],
                "delta": specgap["mean_difference"],
                "cohens_d": specgap["cohens_d"],
                "Mem": result["Mem"],
                "Priv": result["Priv"],
                "Utility": result["Utility"],
                "Agg": result["Agg"],
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "specdiff_tofu.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# SpecDiff · TOFU forget10 · Llama-3.2-1B-Instruct",
        "",
        "超参：lr=5e-5，λ=1.0，β=0.1，κ=0.3，τ=0.02，warmup_steps=1。",
        "",
        "| Seed | SpecGap_f | SpecGap_r | Δ | Cohen's d | Mem | Priv | Utility | Agg |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {fmt(row['SpecGap_f'])} | "
            f"{fmt(row['SpecGap_r'])} | {fmt(row['delta'])} | "
            f"{fmt(row['cohens_d'], 3)} | {fmt(row['Mem'])} | "
            f"{fmt(row['Priv'])} | {fmt(row['Utility'])} | {fmt(row['Agg'])} |"
        )

    if rows:
        keys = (
            "SpecGap_f",
            "SpecGap_r",
            "delta",
            "cohens_d",
            "Mem",
            "Priv",
            "Utility",
            "Agg",
        )
        means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        stds = {
            key: float(np.std([row[key] for row in rows], ddof=1))
            if len(rows) > 1
            else 0.0
            for key in keys
        }
        lines += [
            "",
            "## 跨种子汇总",
            "",
            "| N | SpecGap_f | SpecGap_r | Δ | Cohen's d | Mem | Priv | Utility | Agg |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {len(rows)} | {fmt(means['SpecGap_f'])}±{fmt(stds['SpecGap_f'])} | "
            f"{fmt(means['SpecGap_r'])}±{fmt(stds['SpecGap_r'])} | "
            f"{fmt(means['delta'])}±{fmt(stds['delta'])} | "
            f"{fmt(means['cohens_d'], 3)}±{fmt(stds['cohens_d'], 3)} | "
            f"{fmt(means['Mem'])}±{fmt(stds['Mem'])} | "
            f"{fmt(means['Priv'])}±{fmt(stds['Priv'])} | "
            f"{fmt(means['Utility'])}±{fmt(stds['Utility'])} | "
            f"{fmt(means['Agg'])}±{fmt(stds['Agg'])} |",
            "",
            f"判据：SpecGap_f≈0.33、SpecGap_r<0.10。当前："
            f"{'通过' if means['SpecGap_f'] >= 0.30 and means['SpecGap_r'] < 0.10 else '未通过/待调参'}。",
        ]
    if missing:
        lines += ["", f"待完成 seeds：{', '.join(map(str, missing))}。"]

    markdown_path = RESULTS / "specdiff_tofu.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(
        f"[done] completed={len(rows)}/3 jsonl={jsonl_path} markdown={markdown_path}"
    )


if __name__ == "__main__":
    main()
