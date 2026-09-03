#!/usr/bin/env python3
"""把 results/ou_table3_runs.jsonl 汇总成论文可直接用的 Markdown 表。

产出（默认 results/ou_table3.md）：
  1. 每个方法的 Top-K（按 Agg 降序）+ 与论文靶值的命中判定（±tol）
  2. SimNPO 全 48 点附表：含 HM(Mem, Utility)，用于反推 OU 真实的 model-selection 规律
     （论文说按 HM(Mem, Utility) 选，而 SimNPO 的 Mem 0.32 ≈ Retain 的 0.31，
      暗示实际是「遗忘到 retain 水平即停」）
  3. 未命中 / 缺失项的归因段（不静默跳过）

用法：
    python scripts/ou_table.py                      # 默认只汇总 source=official
    python scripts/ou_table.py --source all         # 含自训
    python scripts/ou_table.py --top 20 --tol 0.02
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime

# 论文 Table 3（用户给定靶值，顺序 Agg / Mem / Priv / Utility）
TARGETS = {
    "SimNPO": {"Agg": 0.53, "Mem": 0.32, "Priv": 0.63, "Utility": 1.00},
    "RMU":    {"Agg": 0.52, "Mem": 0.47, "Priv": 0.50, "Utility": 0.61},
    "Retain": {"Agg": 0.58, "Mem": 0.31, "Priv": 1.00, "Utility": 0.99},
}
DIMS = ["Agg", "Mem", "Priv", "Utility"]

# 官方评测里值得本机重训对拍的超参（论文未给 AltPO 靶值，但 Agg 已高于 SimNPO 靶 0.53）
REPRO_CANDIDATES = [
    {
        "method": "SimNPO",
        "hyper": "lr5e-05_b3.5_a1_d1_g0.125_ep10",
        "repo_id": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.125_ep10",
        "official": {"Mem": 0.3020, "Priv": 0.4880, "Utility": 0.9992, "Agg": 0.4716},
        "note": (
            "官方 48 点最好（未命中论文 0.53/0.32/0.63/1.00，差在 Priv）。"
            "本机复现：tofu_unlearn_one SimNPO + 该串（tag S_p1best），"
            "由 scripts/ou_repro_p1_winners.sh 在 P1 结束后串行开训。"
        ),
    },
    {
        "method": "AltPO",
        "hyper": "lr2e-05_beta0.05_alpha1_epoch10",
        "repo_id": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.05_alpha1_epoch10",
        "official": {"Mem": 0.4284, "Priv": 0.5724, "Utility": 0.9364, "Agg": 0.5826},
        "note": (
            "官方 ckpt 评测 Agg 0.5826，高于论文 SimNPO 靶 0.53 / Retain 靶 0.58。"
            "本机复现：bash community/methods/AltPO/run.sh（已钉死该串；"
            "DPO + alt5_seed_0，lr=2e-5, beta=0.05, alpha=1, ep=10，"
            "双卡 4×4×2、sdpa、adamw_torch）。论文 Table 3 无 AltPO 行，不替代 SimNPO 命中门槛。"
        ),
    },
]

# 已知缺失项的人工归因（不是静默跳过）
KNOWN_GAPS = [
    ("GradAscent", "官方 HF 仓库没有 GradAscent 的 forget10 ckpt：398 个 "
                   "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_* 里不存在该方法"
                   "（分布：AltPO/NPO/RMU/UNDIAL/IdkDPO 各 54、SimNPO 48、GradDiff/IdkNLL 各 40）。"
                   "P1 该格改由 P2 自训补齐。"),
    ("Retain Mem", "论文 0.31 vs 本实现 0.3462（+0.036）：在官方 ES/EM/ParaProb 数值与 HM "
                   "结构下，四个非负分量无法给出 0.31（需要 TR_norm>1）。见 docs/zh/ou-table3-p0.md §三。"),
    ("GradDiff Priv", "论文 3.27e-03：对称映射下过遗忘侧 s(dev=+0.6)≈0.15，与论文不符；"
                      "可用 ou_aggregate.py --s-mia-mode piecewise 做归因实验。"),
    ("Mem HM clamp", "1−norm(*)<0（常见 TR 略高于 init）时夹到 0 且不参与 Mem HM，"
                     "避免报错或整维塌成 0。带 ※ 的行见 jsonl `notes`。"),
]


def hmean(vals):
    vals = [float(v) for v in vals]
    if any(v == 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_runs(path, source=None):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if source and source != "all" and rec.get("source") != source:
                continue
            rows.append(rec)
    return rows


def hits(rec, tol):
    """返回 (是否命中, 每个维度的差值)。无靶值的方法返回 None。"""
    tgt = TARGETS.get(rec["method"])
    if not tgt:
        return None, {}
    diffs = {d: rec[d] - tgt[d] for d in DIMS}
    return all(abs(v) <= tol for v in diffs.values()), diffs


def fmt_row(rec, tol):
    ok, diffs = hits(rec, tol)
    flag = "—" if ok is None else ("✅" if ok else "❌")
    hm_mu = hmean([rec["Mem"], rec["Utility"]])
    mark = " ※" if rec.get("notes") else ""
    src = rec.get("source", "")
    cells = [f"| {src} | `{rec['hyper']}`{mark} "]
    cells += [f"| {rec[d]:.4f} " for d in ["Mem", "Priv", "Utility", "Agg"]]
    cells.append(f"| {hm_mu:.4f} ")
    cells.append(f"| {flag} ")
    if diffs:
        dev = " ".join(f"{d}{diffs[d]:+.2f}" for d in DIMS)
        cells.append(f"| {dev} |")
    else:
        cells.append("| |")
    return "".join(cells)


def main():
    p = argparse.ArgumentParser(description="生成 OU Table 3 四维汇总表")
    p.add_argument("--runs", default="results/ou_table3_runs.jsonl")
    p.add_argument("--out", default="results/ou_table3.md")
    p.add_argument("--top", type=int, default=10, help="每个方法展示的 Top-K")
    p.add_argument("--tol", type=float, default=0.02, help="命中容差")
    p.add_argument("--full-methods", default="SimNPO",
                   help="额外输出全量附表的（逗号分隔）")
    p.add_argument("--source", default="official",
                   choices=["official", "selftrain", "unknown", "all"],
                   help="只汇总该来源（默认 official，避免自训行混进论文对标）")
    args = p.parse_args()

    rows = load_runs(args.runs, source=args.source)
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    full_methods = {m.strip() for m in args.full_methods.split(",") if m.strip()}

    out = []
    out.append("# OU Table 3 复现汇总（TOFU forget10 · Llama-3.2-1B-Instruct）\n")
    out.append(f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    out.append(f"- 评估器 HEAD：`{git_head()}`")
    out.append(f"- 数据来源：`{args.runs}`（{len(rows)} 条，source={args.source}）")
    out.append("- 口径：Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；"
               "Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。"
               "负分量（常见 1−TR）clamp 到 0 且不参与 HM，带 ※")
    out.append(f"- 命中判定：四维全部落在论文靶值 ±{args.tol:.2f} 内\n")

    # ---- 0) 官方每方法最佳（RMU / UNDIAL 等对标用） --------------------------
    off_best = defaultdict(list)
    for r in load_runs(args.runs, source="official"):
        off_best[r["method"]].append(r)
    if off_best:
        out.append("## 〇、官方 ckpt 每方法最佳（按 Agg，写入本表）\n")
        out.append("| 方法 | 超参串 | Mem | Priv | Utility | Agg | 命中 |")
        out.append("|---|---|---|---|---|---|---|")
        for method in ("SimNPO", "RMU", "UNDIAL", "AltPO", "NPO", "GradDiff", "IdkDPO", "IdkNLL"):
            recs = off_best.get(method)
            if not recs:
                out.append(f"| {method} | — | — | — | — | — | 尚未评 |")
                continue
            r = max(recs, key=lambda x: x["Agg"])
            ok, _ = hits(r, args.tol)
            flag = "—" if ok is None else ("✅" if ok else "❌")
            out.append(
                f"| {method} | `{r['hyper']}` | {r['Mem']:.4f} | {r['Priv']:.4f} "
                f"| {r['Utility']:.4f} | {r['Agg']:.4f} | {flag} |"
            )
        out.append("")

    # ---- 1) 各方法 Top-K ----------------------------------------------------
    out.append("## 一、各方法 Top-K（按 Agg 降序）\n")
    for method in sorted(by_method):
        recs = sorted(by_method[method], key=lambda r: -r["Agg"])
        tgt = TARGETS.get(method)
        head = f"### {method}（{len(recs)} 个 ckpt"
        if tgt:
            head += f"，靶值 Agg={tgt['Agg']:.2f} / Mem={tgt['Mem']:.2f} / " \
                    f"Priv={tgt['Priv']:.2f} / Utility={tgt['Utility']:.2f}"
        out.append(head + "）\n")
        out.append("| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in recs[: args.top]:
            out.append(fmt_row(r, args.tol))
        n_hit = sum(1 for r in recs if hits(r, args.tol)[0] is True)
        if tgt and recs:
            out.append(f"\n命中靶值的 ckpt 数：**{n_hit} / {len(recs)}**\n")
        elif not tgt:
            out.append("\n（论文未给该方法靶值，不做命中判定）\n")

    # ---- 2) 全量附表 --------------------------------------------------------
    for method in sorted(full_methods & set(by_method)):
        recs = sorted(by_method[method], key=lambda r: -r["Agg"])
        out.append(f"## 附：{method} 全量 {len(recs)} 点（按 Agg 降序，用于反推 model selection）\n")
        out.append("| # | 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(recs, 1):
            out.append(f"| {i} " + fmt_row(r, args.tol)[1:])
        out.append("")

    # ---- 3) 补算备注（clamp 行；自训 G1/GA1 即使 --source official 也附上） ----
    noted = [r for r in rows if r.get("notes")]
    extra_self = []
    if args.source == "official":
        extra_self = [
            r for r in load_runs(args.runs, source="selftrain") if r.get("notes")
        ]
    if noted or extra_self:
        out.append("## 二、补算备注（※ = 1−norm 为负，clamp 0 且不参与 HM）\n")
        out.append("原先 `1−TR<0` 会 `ValueError`，评完也进不了 jsonl。"
                   "`ou_aggregate.py` 改为 clamp 后由 `ou_backfill_aggregate.sh` 补算。"
                   "下表不进 Top-K 排名逻辑之外的额外口径；自训行仅作对照，不参与官方命中判定。\n")
        out.append("| 来源 | 方法 | 超参串 | Mem | Priv | Utility | Agg | 备注 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(noted + extra_self, key=lambda x: (x.get("source", ""), -x["Agg"])):
            note = "；".join(r["notes"])
            out.append(
                f"| {r.get('source','')} | {r['method']} | `{r['hyper']}` "
                f"| {r['Mem']:.4f} | {r['Priv']:.4f} | {r['Utility']:.4f} | {r['Agg']:.4f} "
                f"| {note} |"
            )
        out.append("")

    # ---- 4) 待复现超参 ------------------------------------------------------
    out.append("## 三、待复现超参（官方评测选出，本机尚未重训）\n")
    out.append("| 方法 | 超参串 | 官方 Agg | Mem | Priv | Utility | HF repo |")
    out.append("|---|---|---|---|---|---|---|")
    for c in REPRO_CANDIDATES:
        o = c["official"]
        out.append(
            f"| {c['method']} | `{c['hyper']}` | {o['Agg']:.4f} "
            f"| {o['Mem']:.4f} | {o['Priv']:.4f} | {o['Utility']:.4f} "
            f"| `{c['repo_id']}` |"
        )
    out.append("")
    for c in REPRO_CANDIDATES:
        out.append(f"- **{c['method']} `{c['hyper']}`**：{c['note']}")
    out.append("")

    # ---- 5) 归因 ------------------------------------------------------------
    out.append("## 四、未命中 / 缺失项归因\n")
    for name, reason in KNOWN_GAPS:
        out.append(f"- **{name}**：{reason}")
    missing = sorted(set(TARGETS) - set(by_method))
    for m in missing:
        out.append(f"- **{m}**：靶值存在但 `{args.runs}` 里还没有该方法的任何记录"
                   f"（需先跑 ou_eval_batch.sh --only {m}，或 P0-3 的 retain90/full 本地评测）")
    out.append("")

    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"已写入 {args.out}（{len(rows)} 条记录，{len(by_method)} 个方法）")


if __name__ == "__main__":
    main()
