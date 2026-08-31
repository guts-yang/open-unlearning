#!/usr/bin/env python3
"""汇总 SimNPO 网格调参的各 trial 指标。

扫描 /root/autodl-tmp/saves/tune/simnpo_forget10/<trial_name> 下：
  * evals/TOFU_SUMMARY.json            —— 终点指标
  * checkpoint-*/evals/TOFU_SUMMARY.json —— 每 epoch 轨迹（单卡轨迹模式才有）

从目录名反解超参，施加 MU>=0.55 硬约束，输出：
  * 按 FQ 降序排序的 markdown 表（含 privleak/TR/Prob/ROUGE 与 Retain 基线偏离诊断）
  * trials.json（机器可读，供 tofu_tune_simnpo.sh 的 a2/b/c/final 阶段挑选 top 配置）
  * 欠遗忘 / 过遗忘 / 达标 分类与最优 trial、最优 epoch 提示

用法:
  python scripts/summarize_tune_trials.py [--root ROOT] [--md OUT.md] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_ROOT = Path("/root/autodl-tmp/saves/tune/simnpo_forget10")

# Retain90 基线（来自 docs/repro.md 与官方 retain90 TOFU_SUMMARY.json）
RETAIN = {
    "forget_quality": 1.0,
    "model_utility": 0.5911246639353407,
    "forget_truth_ratio": 0.63,
    "forget_Q_A_Prob": 0.1161314930615481,
    "forget_Q_A_ROUGE": 0.37907355526383685,
    "privleak": 23.477499995304495,
    "extraction_strength": 0.05895894128954535,
}

METRIC_KEYS = [
    "forget_quality",
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "privleak",
    "extraction_strength",
]

NAME_RE = re.compile(
    r"_SimNPO_b([0-9.]+)_g([0-9.]+)_a([0-9.]+)_d([0-9.]+)_lr([0-9.eE+-]+)_e(\d+)$"
)


def fmt(v, nd=4):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    a = abs(f)
    if a < 1e-3 or a >= 1e3:
        return f"{f:.2e}"
    return f"{f:.{nd}f}"


def parse_name(name: str) -> dict | None:
    m = NAME_RE.search(name)
    if not m:
        return None
    return {
        "beta": float(m.group(1)),
        "gamma": float(m.group(2)),
        "alpha": float(m.group(3)),
        "delta": float(m.group(4)),
        "lr": m.group(5),
        "epochs": int(m.group(6)),
    }


def load_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def classify(endpoint: dict) -> str:
    """返回 欠遗忘 / 过遗忘 / 达标(忘干净) / MU不足 / 未评测。"""
    if not endpoint:
        return "未评测"
    mu = endpoint.get("model_utility")
    fq = endpoint.get("forget_quality")
    if mu is None or fq is None:
        return "未评测"
    if mu < 0.55:
        return "MU不足"
    if fq >= 0.05:
        return "达标(忘干净)"
    prob = endpoint.get("forget_Q_A_Prob") or 0.0
    if prob < 0.05:
        return "过遗忘"
    if prob > 0.2:
        return "欠遗忘"
    return "中间"


def collect(root: Path) -> list[dict]:
    trials = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        params = parse_name(d.name)
        if params is None:
            continue
        endpoint = load_summary(d / "evals" / "TOFU_SUMMARY.json")
        # 轨迹
        traj = []
        best_traj = None
        for ck in sorted(d.glob("checkpoint-*")):
            s = load_summary(ck / "evals" / "TOFU_SUMMARY.json")
            if not s:
                continue
            try:
                step = int(str(ck.name).split("-")[-1])
            except ValueError:
                step = -1
            traj.append({"global_step": step, "metrics": s})
            fq = s.get("forget_quality")
            if fq is not None and (best_traj is None or fq > best_traj["metrics"].get("forget_quality", 0)):
                best_traj = traj[-1]
        params["task_name"] = d.name
        params["has_trajectory"] = bool(traj)
        params["endpoint"] = endpoint
        params["trajectory"] = traj
        params["best_traj_step"] = best_traj["global_step"] if best_traj else None
        params["best_traj_fq"] = (
            best_traj["metrics"].get("forget_quality") if best_traj else None
        )
        params["klass"] = classify(endpoint)
        trials.append(params)
    # 排序：先 MU 达标，再 FQ 降序，再 MU 降序
    def sort_key(t):
        ep = t["endpoint"] or {}
        mu = ep.get("model_utility") or -1
        fq = ep.get("forget_quality") or -1
        ok = 1 if mu >= 0.55 else 0
        return (ok, fq, mu)

    trials.sort(key=sort_key, reverse=True)
    return trials


def md_table(trials: list[dict]) -> str:
    head = (
        "| 排名 | beta | gamma | alpha | delta | lr | epochs | 模式 | FQ↑ | MU↑ | TR | "
        "forget Prob | forget ROUGE | privleak | 诊断 | 最优epoch(FQ) |\n"
        "|------|------|-------|-------|-------|----|--------|------|-----|-----|----|"
        "------------|-------------|----------|------|--------------|"
    )
    rows = [head]
    for i, t in enumerate(trials, 1):
        ep = t["endpoint"] or {}
        mode = "轨迹" if t["has_trajectory"] else "终点"
        best_ep = (
            f"{t['best_traj_step']} (FQ={fmt(t['best_traj_fq'])})"
            if t["has_trajectory"] and t["best_traj_fq"] is not None
            else "—"
        )
        rows.append(
            "| {i} | {beta} | {gamma} | {alpha} | {delta} | {lr} | {epochs} | {mode} | "
            "{fq} | {mu} | {tr} | {prob} | {rouge} | {pl} | {kl} | {be} |".format(
                i=i,
                beta=fmt(t["beta"]),
                gamma=fmt(t["gamma"]),
                alpha=fmt(t["alpha"]),
                delta=fmt(t["delta"]),
                lr=t["lr"],
                epochs=t["epochs"],
                mode=mode,
                fq=fmt(ep.get("forget_quality")),
                mu=fmt(ep.get("model_utility")),
                tr=fmt(ep.get("forget_truth_ratio")),
                prob=fmt(ep.get("forget_Q_A_Prob")),
                rouge=fmt(ep.get("forget_Q_A_ROUGE")),
                pl=fmt(ep.get("privleak")),
                kl=t["klass"],
                be=best_ep,
            )
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--md", type=Path, default=DEFAULT_ROOT / "tune_summary.md")
    ap.add_argument("--json", type=Path, default=DEFAULT_ROOT / "trials.json")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"[warn] root 不存在: {root}")
        return
    trials = collect(root)

    # 输出 markdown
    lines = [
        f"# SimNPO 网格调参汇总（TOFU · Llama-3.2-1B-Instruct · forget10）",
        "",
        f"Retain 基线: MU={fmt(RETAIN['model_utility'])} TR={fmt(RETAIN['forget_truth_ratio'])} "
        f"Prob={fmt(RETAIN['forget_Q_A_Prob'])} ROUGE={fmt(RETAIN['forget_Q_A_ROUGE'])} privleak={fmt(RETAIN['privleak'])}",
        "",
        "硬约束 MU>=0.55；主目标 FQ 最大；里程碑 FQ>=1e-3；理想 FQ>=0.05（统计意义上忘干净）。",
        "",
        "方向： forget 的 Prob/ROUGE 应**趋近** Retain 基线(0.116/0.379)而非压到 0；privleak 趋近 0。",
        "",
        md_table(trials),
        "",
    ]
    # 最优建议
    done = [t for t in trials if t["endpoint"]]
    if done:
        best = done[0]
        lines.append(f"## 建议")
        lines.append(
            f"* 当前最优: beta={best['beta']} gamma={best['gamma']} alpha={best['alpha']} "
            f"delta={best['delta']} lr={best['lr']} epochs={best['epochs']} "
            f"→ FQ={fmt(best['endpoint'].get('forget_quality'))} MU={fmt(best['endpoint'].get('model_utility'))} "
            f"[{best['klass']}]"
        )
        if best["has_trajectory"] and best["best_traj_fq"] is not None:
            lines.append(
                f"* 该配置在 global_step={best['best_traj_step']} 出现 FQ 峰值 "
                f"{fmt(best['best_traj_fq'])}，可在 Stage B 用单卡轨迹锁定停止点。"
            )
    args.md.write_text("\n".join(lines), encoding="utf-8")

    # 输出 json（供后续阶段挑选 top 配置）
    out = []
    for t in done:
        out.append(
            {
                "task_name": t["task_name"],
                "beta": t["beta"],
                "gamma": t["gamma"],
                "alpha": t["alpha"],
                "delta": t["delta"],
                "lr": t["lr"],
                "epochs": t["epochs"],
                "has_trajectory": t["has_trajectory"],
                "klass": t["klass"],
                "metrics": {k: t["endpoint"].get(k) for k in METRIC_KEYS},
                "best_traj_step": t["best_traj_step"],
                "best_traj_fq": t["best_traj_fq"],
            }
        )
    args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"trials={len(trials)} done={len(done)} -> {args.md} / {args.json}")


if __name__ == "__main__":
    main()
