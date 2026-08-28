#!/usr/bin/env python3
"""Upsert a TOFU eval summary into results/<base>.json and regenerate the markdown table."""

from __future__ import annotations

import argparse
import json
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


def write_markdown(store: dict, md_path: Path) -> None:
    model = store["model"]
    lines = [
        f"# TOFU · `{model}`",
        "",
        "本机复现结果（动态表）。同一方法同一 forget split 会被覆盖更新。",
        "",
        f"更新时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
        "## 本机指标",
        "",
        "| 方法 | split | FQ | MU | TR | forget Prob | forget ROUGE | privleak | ES |",
        "|------|-------|----|----|----|-------------|--------------|----------|----|",
    ]
    for r in store["runs"]:
        m = r.get("metrics", {})
        lines.append(
            "| {method} | {split} | {fq} | {mu} | {tr} | {prob} | {rouge} | {priv} | {es} |".format(
                method=r.get("method", ""),
                split=r.get("forget_split", ""),
                fq=fmt(m.get("forget_quality")),
                mu=fmt(m.get("model_utility")),
                tr=fmt(m.get("forget_truth_ratio")),
                prob=fmt(m.get("forget_Q_A_Prob")),
                rouge=fmt(m.get("forget_Q_A_ROUGE")),
                priv=fmt(m.get("privleak")),
                es=fmt(m.get("extraction_strength")),
            )
        )
    lines += [
        "",
        "FQ = forget_quality，MU = model_utility，TR = forget_truth_ratio，ES = extraction_strength。",
        "",
        "## 官方未调参对照（仅 FQ / MU / TR）",
        "",
        "来源：`docs/repro.md`，硬件与 torch 不同，只比量级。",
        "",
        "| 方法 | split | 官方 FQ | 官方 MU | 官方 TR | 本机 FQ | 本机 MU | 本机 TR |",
        "|------|-------|---------|---------|---------|---------|---------|---------|",
    ]
    for r in store["runs"]:
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
        "仓库 `community/leaderboard.md` 的 1B 表没有调参方法，**没有覆盖全部 8 列的公开 SOTA**。下面分两档：",
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
        "论文 Table 3 调参后综合分第一是 SimNPO（Agg 0.53），超参不同，对不上本表 8 列。未调参时 SimNPO 的 FQ 极差，**不是** FQ 的参照最优。",
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
        upsert_run(store, run)

    json_path.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(store, md_path)
    print(f"updated {json_path} and {md_path}")


if __name__ == "__main__":
    main()
