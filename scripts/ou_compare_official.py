#!/usr/bin/env python3
"""P0-3 的守门脚本：把本地基线评测结果逐项对照官方 open-unlearning/eval 日志。

对不上就 exit 1，让批量流程停在 P0（用户要求：对不上先别往下走）。

比对项（官方 summary 里有的 + 只能从 EVAL 取到的）：
    extraction_strength / exact_memorization / model_utility
    mia_loss / mia_zlib / mia_min_k / mia_min_k_plus_plus
    forget_Q_A_PARA_Prob（官方 summary 没有，从 TOFU_EVAL.json 取）
    forget_truth_ratio（closer 版，从 EVAL 取；注意 Mem 用的是现算的 prob_mean 版）

用法：
    python scripts/ou_compare_official.py                       # 用默认路径
    python scripts/ou_compare_official.py --tol 0.03 --model Llama-3.2-1B-Instruct
"""

import argparse
import json
import os
import sys

MODEL = "Llama-3.2-1B-Instruct"

# 官方日志（setup_data.py --eval_logs 下载，保持只读基准）
OFFICIAL_RETAIN = f"saves/eval/tofu_{MODEL}_retain90"
OFFICIAL_FULL = f"saves/eval/tofu_{MODEL}_full/evals_forget10"

COMPARE_KEYS = [
    "extraction_strength",
    "exact_memorization",
    "model_utility",
    "mia_loss",
    "mia_zlib",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "forget_Q_A_PARA_Prob",
    "forget_truth_ratio",
]


def load_pair(d):
    """读 (summary, eval_logs)；文件缺失时返回 (None, None)。"""
    s = os.path.join(d, "TOFU_SUMMARY.json")
    e = os.path.join(d, "TOFU_EVAL.json")
    summary = json.load(open(s)) if os.path.exists(s) else None
    evals = json.load(open(e)) if os.path.exists(e) else None
    return summary, evals


def get_val(summary, evals, key):
    """优先 summary，缺失时从 TOFU_EVAL.json 取 agg_value。"""
    if summary and key in summary:
        return summary[key]
    if evals and key in evals and isinstance(evals[key], dict):
        return evals[key].get("agg_value")
    return None


def compare(label, off_dir, loc_dir, tol):
    off_sum, off_eval = load_pair(off_dir)
    loc_sum, loc_eval = load_pair(loc_dir)
    print(f"\n--- {label} ---")
    print(f"  官方：{off_dir}\n  本地：{loc_dir}")
    if loc_sum is None:
        print(f"  [FAIL] 本地产物缺失：{loc_dir}/TOFU_SUMMARY.json（先跑 ou_eval_baselines.sh）")
        return False

    print(f"  {'指标':<28}{'官方':>12}{'本地':>12}{'差值':>12}  判定")
    ok_all = True
    for k in COMPARE_KEYS:
        o = get_val(off_sum, off_eval, k)
        l = get_val(loc_sum, loc_eval, k)
        if l is None:
            print(f"  {k:<28}{'-':>12}{'MISSING':>12}{'-':>12}  ❌ 本地缺该指标")
            ok_all = False
            continue
        if o is None:
            print(f"  {k:<28}{'-':>12}{l:>12.5f}{'-':>12}  —  官方无对照，跳过")
            continue
        d = l - o
        ok = abs(d) <= tol
        ok_all &= ok
        print(f"  {k:<28}{o:>12.5f}{l:>12.5f}{d:>+12.5f}  {'✅' if ok else '❌'}")

    # 新口径下官方从未评测的两个指标，单独提示（不算失败）
    for k in ["forget_Q_A_gibberish"]:
        v = get_val(loc_sum, loc_eval, k)
        note = f"{v:.5f}（官方从未评测，本地首次产出，作为 Utility 的 Fluency 分母）" \
            if v is not None else "本地也缺失 → Utility 会报错，必须补齐"
        print(f"  {k:<28}{'-':>12}{note}")
        if v is None:
            ok_all = False
    return ok_all


def main():
    p = argparse.ArgumentParser(description="本地基线 vs 官方日志对照")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--tol", type=float, default=0.02, help="绝对容差")
    p.add_argument("--local-retain", default=f"saves/eval/tofu_{MODEL}_retain90_local")
    p.add_argument("--local-full", default=f"saves/eval/tofu_{MODEL}_full_local/evals_forget10")
    p.add_argument("--official-retain", default=OFFICIAL_RETAIN)
    p.add_argument("--official-full", default=OFFICIAL_FULL)
    args = p.parse_args()

    print(f"=== P0-3 基线对照（容差 ±{args.tol}）===")
    r1 = compare("retain90", args.official_retain, args.local_retain, args.tol)
    r2 = compare("init-finetuned (full)", args.official_full, args.local_full, args.tol)

    print("\n=== 结论 ===")
    if r1 and r2:
        print("[PASS] 本地基线与官方日志一致，可以进 P1")
        return 0
    print("[FAIL] 存在不一致。回查顺序：\n"
          "  ① 本地评测是否用改造后的 configs/eval/tofu.yaml（应含 7 个新指标）\n"
          "  ② Mem 的 TR 定义与归一化（docs/zh/ou-table3-p0.md §二.2）\n"
          "  ③ sMIA 映射与 retain90 参考（§二.3）\n"
          "  ④ Fluency 来源（gibberish class_id=0）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
