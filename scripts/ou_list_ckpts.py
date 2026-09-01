#!/usr/bin/env python3
"""生成 OU Table 3 复现（P1）要评估的官方 ckpt 清单。

从 Hugging Face 搜索 `unlearn_tofu_Llama-3.2-1B-Instruct_forget10`，过滤掉 30 个
辅助模型（neg_tofu_*_forget10_pert_*、pos_tofu_*_forget10_bio_*、
pos_tofu_*_forget10_para_* 各 10），按方法归类并按配额抽样，输出 results/ckpt_list.json
供 scripts/ou_eval_batch.sh 消费。

实测（2026-09-01）：搜索命中 428 个仓库 = 398 个 unlearn ckpt + 30 个辅助模型。
398 个的方法分布：AltPO 54 / NPO 54 / RMU 54 / UNDIAL 54 / IdkDPO 54 /
SimNPO 48 / GradDiff 40 / IdkNLL 40（**没有 GradAscent**，见下）。

命名规则（按 `_` 切分）：
    unlearn_tofu_Llama-3.2-1B-Instruct_forget10_<Method>_<超参串>
    -> method = seg[4]，hyper = "_".join(seg[5:])

用法：
    source /root/autodl-tmp/env_hf.sh            # HF_ENDPOINT=hf-mirror.com
    python scripts/ou_list_ckpts.py              # 写 results/ckpt_list.json
    python scripts/ou_list_ckpts.py --dry-run    # 只打印统计，不写文件
    python scripts/ou_list_ckpts.py --quota SimNPO=48,RMU=54 --out /tmp/x.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

MODEL_TAG = "Llama-3.2-1B-Instruct"
SPLIT = "forget10"
SEARCH = f"unlearn_tofu_{MODEL_TAG}_{SPLIT}"
PREFIX = f"unlearn_tofu_{MODEL_TAG}_{SPLIT}_"

# 论文 Table 3 复现配额（用户指定）。约 142 条。
DEFAULT_QUOTA = {
    "SimNPO": 48,
    "RMU": 54,
    "GradDiff": 8,
    "NPO": 8,
    "UNDIAL": 8,
    "GradAscent": 4,
    "IdkNLL": 4,
    "IdkDPO": 4,
    "AltPO": 4,
}

# P2 自训要对齐的官方 ckpt，必须强制入选（否则抽样可能漏掉）
FORCE_INCLUDE = {
    "SimNPO": [
        "lr1e-05_b4.5_a1_d0_g0.125_ep10",   # P2-T1
        "lr2e-05_b3.5_a1_d1_g0.25_ep10",    # P2-T2
    ],
    "GradDiff": [
        "lr1e-05_alpha5_epoch10",           # P2-T3（注意是 epoch10 不是 ep10）
    ],
}


def lr_value(hyper: str) -> float:
    """从超参串里解析学习率数值，用于稳定排序（lr1e-05 -> 1e-05）。"""
    m = re.search(r"lr([0-9eE.+-]+)", hyper)
    return float(m.group(1)) if m else float("inf")


def sort_key(hyper: str):
    """确定性排序：先按 lr 数值，再按原始字符串。"""
    return (lr_value(hyper), hyper)


def evenly_space(seq, need):
    """从 seq 中等间距取 need 个，保证取满（去重后用未取到的下标补齐）。"""
    n = len(seq)
    if need >= n:
        return list(seq)
    if need <= 0:
        return []
    idxs = [round(i * (n - 1) / (need - 1)) for i in range(need)] if need > 1 else [0]
    picked, seen = [], set()
    for i in idxs:
        if i not in seen:
            picked.append(i)
            seen.add(i)
    for i in range(n):
        if len(picked) >= need:
            break
        if i not in seen:
            picked.append(i)
            seen.add(i)
    return [seq[i] for i in sorted(picked[:need])]


def select(hy_list, quota, forced):
    """按配额选择：强制项优先，其余在排序后的列表上等间距抽样。"""
    hy = sorted(hy_list, key=sort_key)
    chosen = [h for h in hy if h in forced]
    rest = [h for h in hy if h not in forced]
    need = max(0, min(quota, len(hy)) - len(chosen))
    return chosen + evenly_space(rest, need)


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def fetch_repo_ids(search):
    """列出 HF 上匹配搜索词的仓库 id（尊重 HF_ENDPOINT）。"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[error] 缺少 huggingface_hub，请先 source 项目 venv", file=sys.stderr)
        sys.exit(2)
    api = HfApi()
    ids = [m.id for m in api.list_models(search=search, limit=1000)]
    if not ids:
        print(
            "[error] 搜索返回空。请确认已 `source /root/autodl-tmp/env_hf.sh`"
            "（HF_ENDPOINT=hf-mirror.com）且网络可达。",
            file=sys.stderr,
        )
        sys.exit(3)
    return ids


def main():
    p = argparse.ArgumentParser(
        description="生成 P1 官方 ckpt 评估清单",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--search", default=SEARCH, help="HF 搜索词")
    p.add_argument("--out", default="results/ckpt_list.json", help="清单输出路径")
    p.add_argument("--quota", default=None,
                   help="覆盖配额，格式 SimNPO=48,RMU=54（未列出的方法用默认）")
    p.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = p.parse_args()

    quota = dict(DEFAULT_QUOTA)
    if args.quota:
        for kv in args.quota.split(","):
            if not kv.strip():
                continue
            k, v = kv.split("=")
            quota[k.strip()] = int(v)

    ids = fetch_repo_ids(args.search)
    names = [i.split("/")[-1] for i in ids]

    unlearn = [n for n in names if n.startswith(PREFIX)]
    aux = [n for n in names if not n.startswith(PREFIX)]

    by_method = defaultdict(list)
    for n in unlearn:
        seg = n.split("_")
        if len(seg) < 6:
            print(f"[warn] 无法解析方法名，跳过：{n}", file=sys.stderr)
            continue
        by_method[seg[4]].append("_".join(seg[5:]))

    methods_meta, entries = {}, []
    for method in sorted(set(list(by_method) + list(quota))):
        avail = by_method.get(method, [])
        q = quota.get(method, 0)
        forced = FORCE_INCLUDE.get(method, [])
        missing_forced = [f for f in forced if f not in avail]
        if missing_forced:
            print(f"[warn] {method} 强制项在官方仓库中不存在：{missing_forced}", file=sys.stderr)
        picked = select(avail, q, forced)
        methods_meta[method] = {
            "available": len(avail),
            "quota": q,
            "selected": len(picked),
            "note": ("官方无该方法的 forget10 ckpt（398 个中不存在）"
                     if not avail else ""),
        }
        for hyper in picked:
            entries.append({
                "repo_id": f"open-unlearning/{PREFIX}{method}_{hyper}",
                "name": f"{PREFIX}{method}_{hyper}",
                "method": method,
                "hyper": hyper,
                "status": "pending",
            })

    total = len(entries)
    print(f"搜索命中 {len(ids)} 个仓库：unlearn ckpt {len(unlearn)}，辅助模型 {len(aux)}")
    if aux:
        kinds = defaultdict(int)
        for n in aux:
            tag = n.split("_")[0] + "_" + ("_".join(n.split("_")[-3:-2]) or "?")
            kinds[tag] += 1
        print(f"  已排除辅助模型：{dict(kinds)}")
    print(f"{'方法':<12}{'可用':>6}{'配额':>6}{'入选':>6}  备注")
    for m, meta in sorted(methods_meta.items(), key=lambda x: -x[1]["selected"]):
        print(f"{m:<12}{meta['available']:>6}{meta['quota']:>6}{meta['selected']:>6}"
              f"  {meta['note']}")
    print(f"合计入选 {total} 条")

    if args.dry_run:
        print("[dry-run] 未写文件")
        return

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "head": git_head(),
        "search": args.search,
        "model": MODEL_TAG,
        "forget_split": SPLIT,
        "matched_unlearn": len(unlearn),
        "matched_aux": len(aux),
        "methods": methods_meta,
        "ckpts": entries,
    }
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
