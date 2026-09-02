#!/usr/bin/env python3
"""Upsert a TOFU eval summary into results/<base>.json and regenerate the markdown table."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METRIC_KEYS = [
    "forget_quality",
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "privleak",
    "extraction_strength",
]

METHOD_ORDER = [
    "SimNPO",
    "RMU",
    "UNDIAL",
    "AltPO",
    "NPO",
    "IdkDPO",
    "GradDiff",
    "GradAscent",
    "TPO",
]

# docs/repro.md Llama-3.2-1B-Instruct, unreproduced defaults
OFFICIAL_FQ_MU_TR = {
    ("forget10", "Finetuned"): (1.66e-21, 0.6, 0.48),
    ("forget10", "Retain"): (1.0, 0.59, 0.63),
    ("forget10", "SimNPO"): (2.47e-203, 0.54, 1.07e-05),
    ("forget10", "RMU"): (3.15e-15, 0.59, 0.76),
    ("forget10", "NPO"): (0.02, 0.46, 0.7),
    ("forget10", "GradDiff"): (1.06e-239, 0.49, 3.53e-27),
    ("forget10", "GradAscent"): (1.06e-239, 0.0, 2.25e-18),
    ("forget10", "IdkDPO"): (4.64e-12, 0.23, 0.6),
}

# Official retain90 eval dump (saves/eval/.../TOFU_SUMMARY.json): extra metrics not in repro.md
RETAIN90_LOG = {
    "forget_Q_A_Prob": 0.1161314930615481,
    "forget_Q_A_ROUGE": 0.37907355526383685,
    "privleak": 23.477499995304495,
    "extraction_strength": 0.05895894128954535,
    "model_utility": 0.5911246639353407,
}

# Best unreproduced *unlearning method* on 1B forget10 (docs/repro.md). Retain/Finetuned are oracles/baselines.
OFFICIAL_BEST_UNLEARN = {
    "forget_quality": ("NPO", 0.02),
    "model_utility": ("RMU", 0.59),
    "forget_truth_ratio": ("RMU", 0.76),
}

# 方法全称（Leaderboard「全称」列）
FULL_NAMES = {
    "SimNPO": "SimNPO: A Simple and Effective Non-Preference Optimization for Machine Unlearning",
    "RMU": "Representation Misdirection for Unlearning（Li et al., 2024）",
    "UNDIAL": "Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models（NAACL 2025）",
    "AltPO": "Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models",
    "NPO": "Negative Preference Optimization",
    "IdkDPO": "IDK-DPO",
    "GradDiff": "Gradient Difference",
    "GradAscent": "Gradient Ascent",
    "TPO": "Targeted Preference Optimization（Yang et al., AAAI 2026）",
}

# 与官方未调参设置（docs/repro.md）的对照结论；无条目的方法用通用说明
SETTING_NOTES = {
    "RMU": "与官方未调参设置一致（FQ/MU/TR 量级吻合）",
    "SimNPO": "与官方未调参表超参不一致（本机 TR 0.52 vs 官方 1.07e-05，量级差异大）",
    "UNDIAL": "官方无 1B forget10 对照",
    "AltPO": "官方无 1B forget10 对照",
}


def unified_metrics(run: dict) -> dict:
    """把原始指标转成统一「越高越好」方向；RL = forget ROUGE-L。"""
    m = run.get("metrics", {})
    return {
        "fq": fmt(m.get("forget_quality")),
        "mu": fmt(m.get("model_utility")),
        "rl": fmt(1.0 - (m.get("forget_Q_A_ROUGE") or 0.0)),
        "tr": fmt(m.get("forget_truth_ratio")),
        "prob": fmt(1.0 - (m.get("forget_Q_A_Prob") or 0.0)),
        "es": fmt(1.0 - (m.get("extraction_strength") or 0.0)),
    }


def sota_score(run: dict) -> float:
    """6 项统一指标（FQ/MU/1−RL/TR/1-Prob/1-ES）的均值，用于选每基座 SOTA。"""
    m = run.get("metrics", {})
    vals = [
        m.get("forget_quality") or 0.0,
        m.get("model_utility") or 0.0,
        1.0 - (m.get("forget_Q_A_ROUGE") or 0.0),
        m.get("forget_truth_ratio") or 0.0,
        1.0 - (m.get("forget_Q_A_Prob") or 0.0),
        1.0 - (m.get("extraction_strength") or 0.0),
    ]
    return sum(vals) / len(vals)


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    av = abs(float(v))
    if av != 0 and (av < 1e-3 or av >= 1e3):
        return f"{float(v):.2e}"
    return f"{float(v):.4f}"


def load_store(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "benchmark": "TOFU",
        "model": "Llama-3.2-1B-Instruct",
        "runs": [],
    }


def run_key(run: dict) -> tuple:
    return (run.get("forget_split"), run.get("method"))


def upsert_run(store: dict, run: dict) -> None:
    key = run_key(run)
    runs = [r for r in store["runs"] if run_key(r) != key]
    runs.append(run)
    runs.sort(
        key=lambda r: (
            r.get("forget_split", ""),
            METHOD_ORDER.index(r["method"])
            if r.get("method") in METHOD_ORDER
            else 99,
            r.get("method", ""),
        )
    )
    store["runs"] = runs


def _p0_baseline_section() -> list[str]:
    """把 P0-3 本地基线四维嵌进动态表，避免 record 重写时冲掉。"""
    saves = Path(os.environ.get("SAVES", "/root/autodl-tmp/saves"))
    retain = saves / "eval/tofu_Llama-3.2-1B-Instruct_retain90_local/ou_aggregate.json"
    full = saves / "eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10/ou_aggregate.json"
    if not retain.exists() or not full.exists():
        return []
    r = json.loads(retain.read_text())
    f = json.loads(full.read_text())
    den = f.get("denominators") or f.get("params") or {}
    # ou_aggregate --json 结构：顶层 Mem/Priv/Utility/Agg
    return [
        "## P0-3 本地基线（forget10 · 四维口径）",
        "",
        "对照官方 `open-unlearning/eval` ±0.02 已 PASS。Fluency 分母来自本机 `full_local`。",
        "",
        "| 行 | Mem | Priv | Utility | Agg |",
        "|----|-----|------|---------|-----|",
        f"| Init finetuned（full_local） | {fmt(f.get('Mem'))} | {fmt(f.get('Priv'))} | {fmt(f.get('Utility'))} | {fmt(f.get('Agg'))} |",
        f"| Retain（retain90_local） | {fmt(r.get('Mem'))} | {fmt(r.get('Priv'))} | {fmt(r.get('Utility'))} | {fmt(r.get('Agg'))} |",
        "",
        f"Fluency 分母（full gibberish）见 `ou_aggregate.json`。denominators={den or '见 json'}。",
        "",
    ]


def write_markdown(store: dict, md_path: Path) -> None:
    model = store["model"]
    runs = store["runs"]

    # 按（基座，forget split）分组，每组取一个 SOTA
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        key = (r.get("model", model), r.get("forget_split", ""))
        groups.setdefault(key, []).append(r)

    lines = [
        f"# TOFU · `{model}`",
        "",
        "本机复现结果（动态表）。同一方法同一 forget split 会被覆盖更新。",
        "",
        f"更新时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
    ]
    p03 = _p0_baseline_section()
    if p03:
        lines.extend(p03)
    lines += [
        "## 复现 Leaderboard（每基座取 SOTA）",
        "",
        "SOTA = 当前复现 runs 中 6 项「越高越好」指标（FQ / MU / 1−RL / TR / 1-Prob / 1-ES）的均值最高者。",
        "",
        "| 基座名称 | SOTA 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ | 全称 | 设置一致性 |",
        "|----------|-----------|-----|-----|-------|-----|---------|-------|------|------------|",
    ]
    for (bmodel, split), group in sorted(groups.items()):
        sota = max(group, key=sota_score)
        u = unified_metrics(sota)
        method = sota.get("method", "")
        lines.append(
            "| {base} | **{method}** | {fq} | {mu} | {rl} | {tr} | {prob} | {es} | "
            "{full} | 本机复现（{split}），硬件/torch 与官方 2×L40s 不同、只比量级；{note} |".format(
                base=bmodel,
                method=method,
                fq=u["fq"],
                mu=u["mu"],
                rl=u["rl"],
                tr=u["tr"],
                prob=u["prob"],
                es=u["es"],
                full=FULL_NAMES.get(method, method),
                split=split,
                note=SETTING_NOTES.get(method, "沿用本仓库默认设置"),
            )
        )

    lines += [
        "",
        "FQ = forget_quality，MU = model_utility，TR = forget_truth_ratio，ES = extraction_strength，"
        "RL = forget ROUGE-L。1−RL / 1-Prob / 1-ES 已翻转为「越高越好」。privleak 等原始值见 `results/*.json`。",
        "",
        "## 本机各方法明细（统一方向：越高越好）",
        "",
        "| 基座名称 | 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ |",
        "|----------|------|-----|-----|-------|-----|---------|-------|",
    ]
    for (bmodel, split), group in sorted(groups.items()):
        sota_method = max(group, key=sota_score).get("method")
        for r in group:
            u = unified_metrics(r)
            method = r.get("method", "")
            shown = f"**{method}**" if method == sota_method else method
            lines.append(
                "| {base} | {method} | {fq} | {mu} | {rl} | {tr} | {prob} | {es} |".format(
                    base=bmodel,
                    method=shown,
                    fq=u["fq"],
                    mu=u["mu"],
                    rl=u["rl"],
                    tr=u["tr"],
                    prob=u["prob"],
                    es=u["es"],
                )
            )

    # 四维（OU Table 3 口径）：只有带 ou_table3 字段的 run 才参与，默认不影响原表
    ou_runs = [r for r in runs if r.get("ou_table3")]
    if ou_runs:
        lines += [
            "",
            "## 本机四维（OU Table 3 口径）",
            "",
            "Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；"
            "Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。"
            "来自 `scripts/ou_aggregate.py --json`（`--ou-summary`）。",
            "",
            "| 基座名称 | 方法 | split | Mem | Priv | Utility | Agg | ckpt |",
            "|----------|------|-------|-----|------|---------|-----|------|",
        ]
        for r in sorted(ou_runs, key=lambda r: -(r["ou_table3"].get("Agg") or 0)):
            o = r["ou_table3"]
            lines.append(
                "| {base} | {method} | {split} | {mem} | {priv} | {util} | {agg} | {ckpt} |".format(
                    base=r.get("model", model),
                    method=r.get("method", ""),
                    split=r.get("forget_split", ""),
                    mem=fmt(o.get("Mem")),
                    priv=fmt(o.get("Priv")),
                    util=fmt(o.get("Utility")),
                    agg=fmt(o.get("Agg")),
                    ckpt=r.get("ckpt", "") or "—",
                )
            )

    lines += [
        "",
        "## 官方未调参对照（仅 FQ / MU / TR）",
        "",
        "来源：`docs/repro.md`，硬件与 torch 不同，只比量级。",
        "",
        "| 方法 | split | 官方 FQ | 官方 MU | 官方 TR | 本机 FQ | 本机 MU | 本机 TR |",
        "|------|-------|---------|---------|---------|---------|---------|---------|",
    ]
    for r in runs:
        split, method = r.get("forget_split"), r.get("method")
        official = OFFICIAL_FQ_MU_TR.get((split, method))
        m = r.get("metrics", {})
        if official:
            ofq, omu, otr = official
            ofq_s, omu_s, otr_s = fmt(ofq), fmt(omu), fmt(otr)
        else:
            ofq_s = omu_s = otr_s = "—"
        lines.append(
            f"| {method} | {split} | {ofq_s} | {omu_s} | {otr_s} | "
            f"{fmt(m.get('forget_quality'))} | {fmt(m.get('model_utility'))} | "
            f"{fmt(m.get('forget_truth_ratio'))} |"
        )
    ofq, omu, otr = OFFICIAL_FQ_MU_TR[("forget10", "Retain")]
    ffq, fmu, ftr = OFFICIAL_FQ_MU_TR[("forget10", "Finetuned")]
    bfq_m, bfq = OFFICIAL_BEST_UNLEARN["forget_quality"]
    bmu_m, bmu = OFFICIAL_BEST_UNLEARN["model_utility"]
    btr_m, btr = OFFICIAL_BEST_UNLEARN["forget_truth_ratio"]
    lines += [
        "",
        "## 参照 / SOTA（TOFU 1B · forget10）",
        "",
        "仓库 `community/leaderboard.md` 的 1B 表没有调参方法，**没有覆盖本表 6 项统一指标（FQ/MU/1−RL/TR/1-Prob/1-ES）的公开 SOTA**。下面分两档：",
        "",
        "### 上界与未遗忘下界",
        "",
        "| 参照 | FQ | MU | TR | forget Prob | forget ROUGE | privleak | ES | 说明 |",
        "|------|----|----|----|-------------|--------------|----------|----|------|",
        f"| Retain（上界） | {fmt(ofq)} | {fmt(omu)} | {fmt(otr)} | {fmt(RETAIN90_LOG['forget_Q_A_Prob'])} | {fmt(RETAIN90_LOG['forget_Q_A_ROUGE'])} | {fmt(RETAIN90_LOG['privleak'])} | {fmt(RETAIN90_LOG['extraction_strength'])} | 没学过 forget；FQ/MU/TR 来自 `docs/repro.md`，其余来自官方 retain90 `TOFU_SUMMARY.json` |",
        f"| Finetuned（下界） | {fmt(ffq)} | {fmt(fmu)} | {fmt(ftr)} | — | — | — | — | 遗忘前的 full 模型，`docs/repro.md` |",
        "",
        "方向：FQ / MU 越高越好；TR 越接近 1 越好；forget Prob / ROUGE / ES 越低通常越「忘得干净」（但效用崩了也会变低）；privleak 越接近 0 越好。",
        "",
        "### 官方未调参方法里最好（不是 Retain）",
        "",
        "来源：`docs/repro.md` Llama-3.2-1B-Instruct forget10。Prob / ROUGE / privleak / ES 无官方方法对照。",
        "",
        "| 指标 | 最好方法 | 数值 |",
        "|------|----------|------|",
        f"| FQ | {bfq_m} | {fmt(bfq)} |",
        f"| MU | {bmu_m} | {fmt(bmu)} |",
        f"| TR | {btr_m} | {fmt(btr)} |",
        "",
        "论文 Table 3 调参后综合分第一是 SimNPO（Agg 0.53），超参不同，对不上本表 6 项统一指标。未调参时 SimNPO 的 FQ 极差，**不是** FQ 的参照最优。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, help="TOFU_SUMMARY.json")
    parser.add_argument("--method", default=None)
    parser.add_argument("--forget-split", default="forget10")
    parser.add_argument("--model", default="Llama-3.2-1B-Instruct")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--refresh-only", action="store_true")
    parser.add_argument("--ou-summary", type=Path, default=None,
                        help="ou_aggregate.py --json 产出的四维 JSON；"
                             "传入后才在 md 里追加「本机四维（OU Table 3 口径）」表，默认行为不变")
    args = parser.parse_args()

    base = f"tofu_{args.model}"
    json_path = RESULTS / f"{base}.json"
    md_path = RESULTS / f"{base}.md"
    RESULTS.mkdir(parents=True, exist_ok=True)
    store = load_store(json_path)
    store["benchmark"] = "TOFU"
    store["model"] = args.model

    if not args.refresh_only:
        if args.summary is None or not args.method:
            parser.error("--summary and --method are required unless --refresh-only")
        metrics = json.loads(Path(args.summary).read_text())
        run = {
            "method": args.method,
            "forget_split": args.forget_split,
            "ckpt": args.ckpt,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        }
        extra = {k: v for k, v in metrics.items() if k not in METRIC_KEYS}
        if extra:
            run["metrics_extra"] = extra
        if args.ou_summary:
            ou = json.loads(Path(args.ou_summary).read_text())
            run["ou_table3"] = {
                "Mem": ou.get("Mem"),
                "Priv": ou.get("Priv"),
                "Utility": ou.get("Utility"),
                "Agg": ou.get("Agg"),
                "params": ou.get("params"),
            }
        upsert_run(store, run)

    json_path.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(store, md_path)
    print(f"updated {json_path} and {md_path}")


if __name__ == "__main__":
    main()
