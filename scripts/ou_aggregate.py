#!/usr/bin/env python3
"""OU Table 3 四维聚合脚本（TOFU forget10 口径）。

输入 TOFU_SUMMARY.json（以及同目录 TOFU_EVAL.json），按 OpenUnlearning 论文 §F.1 输出
Mem / Priv / Utility / Agg 四维数值，并打印归一化分母（init-finetuned 各指标值）与
sMIA 参考（retain90 各 MIA AUC），供人工核对。

口径（论文 §F.1，已用官方 open-unlearning/eval 数据校准，见 docs/zh/ou-table3-p0.md）：

    Mem     = HM(1 - norm(ES), 1 - norm(EM), 1 - norm(Para.Prob), 1 - norm(TR))
              若某分量因 target>init 略负（常见 TR_norm>1），夹到 0 且不参与 HM，
              并在 result.notes 备注；避免 hmean 报错或因 0 把整维打成 0。
    Priv    = HM(s_MIA(loss), s_MIA(zlib), s_MIA(min_k), s_MIA(min_k++))
    Utility = HM(norm(MU), norm(Forget Fluency))
    Agg     = HM(Mem, Priv, Utility)          # 论文 Table 3（含 Priv 的三维 Agg）

    norm(x) = x_target / x_init_finetuned     # 全部按 init-finetuned 模型归一化
    TR      = OU 定义 truth ratio（prob_mean 聚合）:
              mean( correct / (correct + wrong) ) = mean( 1 / (1 + score) )
              其中 score = wrong / correct，来自 TOFU_EVAL.json 中
              forget_truth_ratio.value_by_index[*].score（逐样本原始值）。
              注意：TOFU_SUMMARY.json 里的 forget_truth_ratio 是 TOFU 原版
              closer_to_1_better 聚合（不确定度方向），与 OU 论文 Mem 使用的
              prob_mean 版（记忆度方向）不同，不能直接用于 Mem。
    s_MIA   = 1 / (1 + theta * |dev|),  theta = 9.05
              dev = (1 - auc_target - (1 - auc_retain)) / (1 - auc_retain)
                   = (auc_retain - auc_target) / (1 - auc_retain)
              （privleak 风格相对偏差，参照 src/evals/metrics/privacy.py）
              theta 由官方锚点反解：Init finetuned -> Priv = 0.10，Retain -> 1.00。

用法：
    python scripts/ou_aggregate.py <target_summary.json> [options]

选项：
    --init-summary  PATH   init-finetuned（full）模型 summary，默认 saves/eval/.../full/evals_forget10
    --retain-summary PATH  retain90 模型 summary（sMIA 参考），默认 saves/eval/.../retain90
    --s-mia-theta   FLOAT sMIA 映射参数 theta（默认 9.05）
    --s-mia-mode    STR   symmetric（默认，1/(1+theta*|dev|)，与 e9e9ab2 校准一致）
                          或 piecewise（dev>0 过遗忘侧用 exp(-theta*dev)，仅归因实验用）
    --assume-fluency FLOAT 缺 forget_Q_A_gibberish 时用该值占位（默认 None=报错），仅供校准
    --json          PATH  额外把完整结果（四维 + 分量 + 分母 + sMIA 参考）写成 JSON，供批量落盘
"""

import argparse
import json
import os


# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
MIA_ATTACKS = ["mia_loss", "mia_zlib", "mia_min_k", "mia_min_k_plus_plus"]

MODEL_NAME = "tofu_Llama-3.2-1B-Instruct"
DEFAULT_INIT_SUMMARY = (
    f"saves/eval/{MODEL_NAME}_full/evals_forget10/TOFU_SUMMARY.json"
)
DEFAULT_RETAIN_SUMMARY = f"saves/eval/{MODEL_NAME}_retain90/TOFU_SUMMARY.json"

# 由官方锚点反解（见 docs/zh/ou-table3-p0.md）：
#   s_MIA(dev_init) = 0.10  ->  theta ~= 9.05（四攻击 dev 平均）
S_MIA_THETA = 9.05


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def hmean(values):
    """调和平均。按论文 HM 语义：含 0 -> 0；含负值 -> 报错（避免静默错误）。"""
    vals = [float(v) for v in values]
    if any(v < 0 for v in vals):
        raise ValueError(f"Harmonic mean 含负值: {vals}")
    if any(v == 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def hmean_drop_neg(labeled):
    """HM，但把负分量夹到 0 后剔除（不参与平均）。

    评测噪声下 TR/ES 可能略高于 init，使 1-norm < 0。若把 0 留在 HM 里，
    整维直接变 0，比真实遗忘水平差一个数量级。剔除后用其余非负分量平均，
    并返回备注列表。全为负则分数为 0。
    """
    notes = []
    used = []
    for label, raw in labeled:
        v = float(raw)
        if v < 0:
            notes.append(
                f"{label} raw={v:.6f}<0 → clamp 0 且不参与 HM（target 略高于 init）"
            )
            continue
        used.append(v)
    if not used:
        notes.append("全部 HM 分量被 clamp 剔除，分数记 0")
        return 0.0, notes
    return hmean(used), notes


def load_json(path, what):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{what} 文件不存在: {path}")
    with open(path) as f:
        return json.load(f)


def norm(target, init, name):
    """按 init-finetuned 归一化：x_target / x_init。"""
    init = float(init)
    if init <= 0:
        raise ValueError(f"{name}: init-finetuned 值为 {init}，无法作归一化分母")
    return float(target) / init


def get_val(summary, eval_logs, key):
    """取指标 agg_value：优先 summary，缺失时从同目录 TOFU_EVAL.json fallback。

    官方 open-unlearning/eval 的 TOFU_SUMMARY.json 只含顶层指标（如
    forget_Q_A_PARA_Prob 不在其中），但 TOFU_EVAL.json 含所有已计算条目。
    本地用改造后 configs/eval/tofu.yaml 评测时 summary 会包含全部所需 key。
    """
    if summary is not None and key in summary:
        return summary[key]
    if eval_logs is not None and key in eval_logs:
        return eval_logs[key].get("agg_value")
    return None


def tr_ou(eval_logs):
    """OU 论文 truth ratio（prob_mean）：mean(1 / (1 + score))，score = wrong/correct。

    从 TOFU_EVAL.json 的 forget_truth_ratio.value_by_index 逐样本 score 计算。
    若日志中没有该条目（例如官方 summary 之外的纯 summary），返回 None 并打印告警。
    """
    entry = eval_logs.get("forget_truth_ratio")
    if not entry:
        return None
    vbi = entry.get("value_by_index") or {}
    scores = [
        e["score"] for e in vbi.values() if isinstance(e, dict) and e.get("score") is not None
    ]
    if not scores:
        return None
    return sum(1.0 / (1.0 + s) for s in scores) / len(scores)


def s_mia(auc_target, auc_retain, theta=S_MIA_THETA, mode="symmetric"):
    """sMIA：privleak 风格相对偏差映射到 [0,1]（与 retain 金标准的接近度）。

    dev = (auc_retain - auc_target) / (1 - auc_retain)
      dev < 0：泄露侧（target 的 AUC 高于 retain，记忆残留）
      dev > 0：过遗忘侧（target 的 AUC 低于 retain）
    s   = 1 / (1 + theta * |dev|)          # mode=symmetric（默认）
    恒等性：target == retain 时 dev = 0 -> s = 1（Retain -> Priv = 1.00）。

    mode=piecewise（仅归因实验，非默认）：过遗忘侧 dev>0 改用 exp(-theta*dev)，
    惩罚更重，用于验证 GradDiff 这类过遗忘方法能否对齐论文的极低 Priv 值。
    """
    ar = float(auc_retain)
    at = float(auc_target)
    if ar >= 1.0:  # retain AUC = 1 时除数为 0，退化处理
        return 0.0 if at < 1.0 else 1.0
    dev = (ar - at) / (1.0 - ar)
    if mode == "piecewise" and dev > 0:
        import math
        return math.exp(-theta * dev)
    return 1.0 / (1.0 + theta * abs(dev))


# ----------------------------------------------------------------------------
# 主聚合
# ----------------------------------------------------------------------------
def aggregate(target, init_sum, retain_sum, target_eval, init_eval, retain_eval,
              theta=S_MIA_THETA, assume_fluency=None, s_mia_mode="symmetric"):
    """计算四维分数，返回 (result, denominators, retain_refs, fluency)。"""
    # -- 归一化分母（init-finetuned 各指标值，供人工核对） ---------------------
    denominators = {
        "ES": get_val(init_sum, init_eval, "extraction_strength"),
        "EM": get_val(init_sum, init_eval, "exact_memorization"),
        "ParaProb": get_val(init_sum, init_eval, "forget_Q_A_PARA_Prob"),
        "TR_ou": tr_ou(init_eval) if init_eval else None,
        "MU": get_val(init_sum, init_eval, "model_utility"),
        "Fluency": get_val(init_sum, init_eval, "forget_Q_A_gibberish"),
    }
    for k, v in denominators.items():
        if v is None:
            print(f"[warn] init-finetuned 缺少归一化分母 {k}")

    def g(summary, eval_logs, key):
        v = get_val(summary, eval_logs, key)
        if v is None:
            raise KeyError(f"缺少 {key}（summary 与 eval 日志中均无）")
        return v

    # -- Mem -----------------------------------------------------------------
    es_n = norm(g(target, target_eval, "extraction_strength"),
                g(init_sum, init_eval, "extraction_strength"), "ES")
    em_n = norm(g(target, target_eval, "exact_memorization"),
                g(init_sum, init_eval, "exact_memorization"), "EM")
    pp_n = norm(g(target, target_eval, "forget_Q_A_PARA_Prob"),
                g(init_sum, init_eval, "forget_Q_A_PARA_Prob"), "ParaProb")
    tr_t = tr_ou(target_eval)
    tr_i = denominators["TR_ou"]
    if tr_t is None or tr_i is None:
        raise ValueError(
            "TR(OU, prob_mean) 需要 TOFU_EVAL.json（forget_truth_ratio.value_by_index），"
            "请确认提供了 target 与 init-finetuned 的 TOFU_EVAL.json"
        )
    tr_n = norm(tr_t, tr_i, "TR")

    notes = []
    mem, mem_notes = hmean_drop_neg([
        ("1-norm(ES)", 1 - es_n),
        ("1-norm(EM)", 1 - em_n),
        ("1-norm(ParaProb)", 1 - pp_n),
        ("1-norm(TR)", 1 - tr_n),
    ])
    notes.extend(mem_notes)

    # -- Priv -----------------------------------------------------------------
    retain_refs = {}
    for a in MIA_ATTACKS:
        v = get_val(retain_sum, retain_eval, a)
        if v is None:
            raise KeyError(f"retain90 缺少 {a} 参考 AUC（Priv 必须的 sMIA 参考）")
        retain_refs[a] = v
    priv = hmean([
        s_mia(g(target, target_eval, a), retain_refs[a], theta, s_mia_mode)
        for a in MIA_ATTACKS
    ])

    # -- Utility ---------------------------------------------------------------
    mu_n = norm(g(target, target_eval, "model_utility"),
                g(init_sum, init_eval, "model_utility"), "MU")
    flu_t = get_val(target, target_eval, "forget_Q_A_gibberish")
    flu_i = denominators["Fluency"]
    if flu_t is not None and flu_i is not None:
        flu_n = norm(flu_t, flu_i, "Fluency")
        fluency = {"target": flu_t, "init": flu_i}
    elif assume_fluency is not None:
        flu_n = float(assume_fluency)
        fluency = {"assumed": flu_n, "reason": "缺少 forget_Q_A_gibberish，使用 --assume-fluency"}
    else:
        raise ValueError(
            "缺少 forget_Q_A_gibberish（Fluency），请先运行完整评测或提供 --assume-fluency"
        )
    util, util_notes = hmean_drop_neg([
        ("norm(MU)", mu_n),
        ("norm(Fluency)", flu_n),
    ])
    notes.extend(util_notes)

    # -- Agg（论文 Table 3 三维：HM(Mem, Priv, Utility)） ------------------------
    agg, agg_notes = hmean_drop_neg([
        ("Mem", mem),
        ("Priv", priv),
        ("Utility", util),
    ])
    notes.extend(agg_notes)

    result = {
        "Mem": mem,
        "Priv": priv,
        "Utility": util,
        "Agg": agg,
        "notes": notes,
        "components": {
            "mem": {
                "1 - norm(ES)": 1 - es_n,
                "1 - norm(EM)": 1 - em_n,
                "1 - norm(ParaProb)": 1 - pp_n,
                "1 - norm(TR)": 1 - tr_n,
            },
            "priv": {
                a: s_mia(g(target, target_eval, a), retain_refs[a], theta, s_mia_mode)
                for a in MIA_ATTACKS
            },
            "utility": {"norm(MU)": mu_n, "norm(Fluency)": flu_n},
        },
        "denominators": denominators,
        "retain_refs": retain_refs,
        "fluency": fluency,
        "params": {
            "s_mia_theta": theta,
            "s_mia_mode": s_mia_mode,
            "neg_hm_policy": "clamp0_drop",
        },
    }
    return result, denominators, retain_refs, fluency


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="OU Table 3 四维聚合（Mem/Priv/Utility/Agg）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("target_summary", help="目标模型的 TOFU_SUMMARY.json")
    p.add_argument("--init-summary", default=DEFAULT_INIT_SUMMARY,
                   help="init-finetuned（full）模型 TOFU_SUMMARY.json")
    p.add_argument("--retain-summary", default=DEFAULT_RETAIN_SUMMARY,
                   help="retain90 模型 TOFU_SUMMARY.json（sMIA 参考）")
    p.add_argument("--s-mia-theta", type=float, default=S_MIA_THETA,
                   help="sMIA 映射参数 theta")
    p.add_argument("--s-mia-mode", choices=["symmetric", "piecewise"], default="symmetric",
                   help="sMIA 映射模式（默认 symmetric，与 e9e9ab2 校准一致）")
    p.add_argument("--assume-fluency", type=float, default=None,
                   help="缺 forget_Q_A_gibberish 时的 Fluency 占位值（仅供校准）")
    p.add_argument("--json", dest="json_out", default=None,
                   help="额外把完整结果写成 JSON（供批量流程落盘）")
    return p


def main():
    args = build_parser().parse_args()

    init_sum = load_json(args.init_summary, "init-finetuned summary")
    retain_sum = load_json(args.retain_summary, "retain90 summary")
    target = load_json(args.target_summary, "target summary")

    def sibling_eval(summary_path):
        d = os.path.dirname(summary_path)
        return os.path.join(d, "TOFU_EVAL.json") if d else None

    init_eval = load_json(sibling_eval(args.init_summary), "init-finetuned eval")
    retain_eval = load_json(sibling_eval(args.retain_summary), "retain90 eval")
    target_eval = load_json(sibling_eval(args.target_summary), "target eval")

    result, denominators, retain_refs, fluency = aggregate(
        target, init_sum, retain_sum, target_eval, init_eval, retain_eval,
        theta=args.s_mia_theta, assume_fluency=args.assume_fluency,
        s_mia_mode=args.s_mia_mode,
    )

    if args.json_out:
        payload = dict(result)
        payload["inputs"] = {
            "target_summary": args.target_summary,
            "init_summary": args.init_summary,
            "retain_summary": args.retain_summary,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[json] 已写入 {args.json_out}")

    print("\n================ OU Table 3 四维结果 ================")
    for k in ["Mem", "Priv", "Utility", "Agg"]:
        print(f"  {k:<8} = {result[k]:.4f}")
    print("\n------ 分量明细 ------")
    for k, v in result["components"]["mem"].items():
        print(f"  Mem: {k:<18} = {v:.4f}")
    for k, v in result["components"]["priv"].items():
        print(f"  Priv: {k:<14} = {v:.4f}")
    for k, v in result["components"]["utility"].items():
        print(f"  Utility: {k:<14} = {v:.4f}")
    print("\n------ 归一化分母（init-finetuned 各指标值） ------")
    for k, v in denominators.items():
        print(f"  {k:<10} = {v if v is None else f'{v:.5f}'}")
    print("\n------ sMIA 参考（retain90 各 MIA AUC） ------")
    for k, v in retain_refs.items():
        print(f"  {k:<16} = {v:.5f}")
    if result.get("notes"):
        print("\n------ 备注（负分量 clamp） ------")
        for n in result["notes"]:
            print(f"  * {n}")
    print(f"\n  Fluency: target={fluency}")
    print(f"  sMIA theta = {args.s_mia_theta}")


if __name__ == "__main__":
    main()
