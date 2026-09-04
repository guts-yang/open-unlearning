#!/usr/bin/env python3
"""Summarize all SpecDiff TOFU-1B runs into results/specdiff_tofu.md."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SAVES = Path(os.environ.get("SAVES", "/root/autodl-tmp/saves"))
RESULTS = ROOT / "results"
SKIP_TAGS = {"smoke_s0"}


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def hmean(values):
    vals = [float(v) for v in values]
    if any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def hyper_key(cfg):
    return (
        f"lr{cfg['lr']:g}_lam{cfg['lam']:g}_b{cfg['beta']:g}"
        f"_k{cfg['kappa']:g}_t{cfg['tau']:g}_ep{cfg['epochs']:g}"
    )


def load_hparams(ckpt: Path):
    config_path = ckpt / ".hydra" / "config.yaml"
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    method = cfg["trainer"]["method_args"]
    args = cfg["trainer"]["args"]
    return {
        "lr": float(args.get("learning_rate", 5e-5)),
        "lam": float(method["lam"]),
        "beta": float(method["beta"]),
        "kappa": float(method["kappa"]),
        "tau": float(method["tau"]),
        "epochs": int(args.get("num_train_epochs", 10)),
        "warmup_steps": int(method.get("warmup_steps", 1)),
    }


def discover_rows():
    rows = []
    root = SAVES / "unlearn"
    if not root.is_dir():
        return rows
    pattern = re.compile(r"^tofu_1B_SpecDiff_forget10_(.+)$")
    for ckpt in sorted(root.iterdir()):
        match = pattern.match(ckpt.name)
        if not match:
            continue
        tag = match.group(1)
        if tag in SKIP_TAGS:
            continue
        aggregate_path = ckpt / "evals" / "ou_aggregate.json"
        if not aggregate_path.is_file():
            continue
        result = json.loads(aggregate_path.read_text())
        specgap = result["SpecGap"]
        hparams = load_hparams(ckpt)
        mem = float(result["Mem"])
        utility = float(result["Utility"])
        seed_match = re.search(r"seed(\d+)$", tag)
        rows.append(
            {
                "tag": tag,
                "seed": int(seed_match.group(1)) if seed_match else None,
                "checkpoint": str(ckpt),
                "hyper": hyper_key(hparams),
                **hparams,
                "SpecGap_f": specgap["forget"]["mean"],
                "SpecGap_r": specgap["retain"]["mean"],
                "delta": specgap["mean_difference"],
                "cohens_d": specgap["cohens_d"],
                "Mem": mem,
                "Priv": float(result["Priv"]),
                "Utility": utility,
                "Agg": float(result["Agg"]),
                "HM_MU": hmean([mem, utility]),
            }
        )
    return rows


def summarize(group):
    keys = (
        "SpecGap_f",
        "SpecGap_r",
        "delta",
        "cohens_d",
        "Mem",
        "Priv",
        "Utility",
        "Agg",
        "HM_MU",
    )
    means = {key: float(np.mean([row[key] for row in group])) for key in keys}
    stds = {
        key: float(np.std([row[key] for row in group], ddof=1))
        if len(group) > 1
        else 0.0
        for key in keys
    }
    return means, stds


def main():
    rows = discover_rows()
    groups = defaultdict(list)
    for row in rows:
        groups[row["hyper"]].append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS / "specdiff_tofu.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# SpecDiff · TOFU forget10 · Llama-3.2-1B-Instruct",
        "",
        "固定：`warmup_steps=1`，draft = `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`。",
        "模型选择复用 OpenUnlearning 论文 §F.2：按 **HM(Mem, Utility)** 取最高超参（Priv 不进选择）。",
        "Table 3 的 Agg=HM(Mem, Priv, Utility) 仅作展示。Utility 的 Fluency 分母缺失时按 1 处理。",
        "",
        "## 超参组对照（跨 seed 均值，按 HM(Mem,Utility) 降序）",
        "",
        "| 选择 | 超参串 | N | SpecGap_f | SpecGap_r | Δ | d | Mem | Priv | Utility | Agg | HM(Mem,Utility) |",
        "|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    ranked = []
    for hyper, group in groups.items():
        means, stds = summarize(group)
        ranked.append((means["HM_MU"], hyper, group, means, stds))
    ranked.sort(reverse=True)

    selected_hyper = ranked[0][1] if ranked else None
    for hm_mu, hyper, group, means, stds in ranked:
        mark = "**选中**" if hyper == selected_hyper else ""
        lines.append(
            f"| {mark} | `{hyper}` | {len(group)} | "
            f"{fmt(means['SpecGap_f'])}±{fmt(stds['SpecGap_f'])} | "
            f"{fmt(means['SpecGap_r'])}±{fmt(stds['SpecGap_r'])} | "
            f"{fmt(means['delta'])}±{fmt(stds['delta'])} | "
            f"{fmt(means['cohens_d'], 3)}±{fmt(stds['cohens_d'], 3)} | "
            f"{fmt(means['Mem'])}±{fmt(stds['Mem'])} | "
            f"{fmt(means['Priv'])}±{fmt(stds['Priv'])} | "
            f"{fmt(means['Utility'])}±{fmt(stds['Utility'])} | "
            f"{fmt(means['Agg'])}±{fmt(stds['Agg'])} | "
            f"{fmt(means['HM_MU'])}±{fmt(stds['HM_MU'])} |"
        )

    if selected_hyper:
        lines += [
            "",
            f"**选中超参：** `{selected_hyper}`（HM(Mem, Utility) 最高）。",
            "判据 SpecGap_f≥0.30 且 SpecGap_r<0.10：两组 retain 均约 0.30，**均未过 retain 门控**。",
        ]

    for _, hyper, group, means, stds in ranked:
        sample = group[0]
        lines += [
            "",
            f"## `{hyper}`",
            "",
            (
                f"λ={sample['lam']:g}，β={sample['beta']:g}，κ={sample['kappa']:g}，"
                f"τ={sample['tau']:g}，lr={sample['lr']:g}，"
                f"epochs={sample['epochs']}，warmup_steps={sample['warmup_steps']}。"
            ),
            "",
            "| Seed | SpecGap_f | SpecGap_r | Δ | Cohen's d | Mem | Priv | Utility | Agg | HM(Mem,Utility) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(group, key=lambda item: (item["seed"] is None, item["seed"])):
            seed = "—" if row["seed"] is None else row["seed"]
            lines.append(
                f"| {seed} | {fmt(row['SpecGap_f'])} | {fmt(row['SpecGap_r'])} | "
                f"{fmt(row['delta'])} | {fmt(row['cohens_d'], 3)} | {fmt(row['Mem'])} | "
                f"{fmt(row['Priv'])} | {fmt(row['Utility'])} | {fmt(row['Agg'])} | "
                f"{fmt(row['HM_MU'])} |"
            )
        lines += [
            "",
            "| N | SpecGap_f | SpecGap_r | Δ | d | Mem | Priv | Utility | Agg | HM(Mem,Utility) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {len(group)} | {fmt(means['SpecGap_f'])}±{fmt(stds['SpecGap_f'])} | "
                f"{fmt(means['SpecGap_r'])}±{fmt(stds['SpecGap_r'])} | "
                f"{fmt(means['delta'])}±{fmt(stds['delta'])} | "
                f"{fmt(means['cohens_d'], 3)}±{fmt(stds['cohens_d'], 3)} | "
                f"{fmt(means['Mem'])}±{fmt(stds['Mem'])} | "
                f"{fmt(means['Priv'])}±{fmt(stds['Priv'])} | "
                f"{fmt(means['Utility'])}±{fmt(stds['Utility'])} | "
                f"{fmt(means['Agg'])}±{fmt(stds['Agg'])} | "
                f"{fmt(means['HM_MU'])}±{fmt(stds['HM_MU'])} |"
            ),
        ]

    lines += [
        "",
        "## 网格搜索怎么做（§F.2）",
        "",
        "选择规则与 OU 相同：**每个超参只跑 seed=0**，按 **HM(Mem, Utility)** 排序取最高点，再只对中选点补 seed 1/2。",
        "Priv / SpecGap 只记录，不进选择。固定 `warmup_steps=1`、`τ=0.02`、draft=full。",
        "清单见 `scripts/specdiff_grid.yaml`（12 点；其中 1 点已完成，排除已失败的 λ=3/β=0.5）。",
        "",
        "### Stage A · lr × epoch（OU 标准轴，λ=1 β=0.1 κ=0.3）",
        "",
        "| lr | epoch | 说明 |",
        "|---:|---:|---|",
        "| 1e-5 | 5, 10 | 更稳、更少伤 retain |",
        "| 2e-5 | 5, 10 | 中间档 |",
        "| 5e-5 | 5 | 现用 lr 的短训；现用 10 epoch 已完成 |",
        "| 5e-5 | 10 | **已有 3 seed，搜索时跳过** |",
        "",
        "### Stage B · retain 单因子（钉死 lr=5e-5 ep=10 κ=0.3）",
        "",
        "| 变动 | 取值 | 为什么 |",
        "|---|---|---|",
        "| λ | 0.5, 2 | 3 已经失败；只探更弱/略强 TV |",
        "| β | 0, 0.3 | 0=关掉 KL；0.5 已失败，只探 0.3 |",
        "",
        "### Stage C · forget 封顶（钉死 lr=5e-5 ep=10 λ=1 β=0.1）",
        "",
        "| κ | 含义 |",
        "|---:|---|",
        "| 0.2 | overlap 降到 0.8 就停压 forget |",
        "| 0.5 | 允许 overlap 降到 0.5 |",
        "",
        "共 **11 个新点 × seed=0**。不要做 lr×λ×β×κ×epoch 全笛卡尔（162 点）。",
        "跑法：`python scripts/specdiff_grid.py` 看清单；确认后再 `python scripts/specdiff_grid.py --run`。",
        "",
        "## 是否继续调参",
        "",
        "当前两档里按 HM 选 **λ=1、β=0.1、lr=5e-5、ep=10**。不要再把 λ/β 往 3/0.5 推。",
        "若要搜，只跑上面 Stage A–C 的 seed=0 网格，用 HM(Mem, Utility) 重选后再补种子。",
        "",
    ]

    markdown_path = RESULTS / "specdiff_tofu.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"[done] groups={len(groups)} runs={len(rows)} markdown={markdown_path}")


if __name__ == "__main__":
    main()
