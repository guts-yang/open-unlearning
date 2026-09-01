#!/usr/bin/env python3
"""把一次四维结果追加到 results/ou_table3_runs.jsonl（官方 ckpt 评估与自训共用）。

为什么单独抽成脚本：`scripts/ou_eval_one.sh`（官方 ckpt 免训评估）与
`community/methods/TPO/run.sh`（自训）都要把四维结果落到同一个 jsonl，
两处各内联一份 python 迟早漂移。

两种来源的 name 格式不同，解析规则也不同：

- 官方 ckpt：`unlearn_tofu_Llama-3.2-1B-Instruct_forget10_<Method>_<超参串>`
  → method/hyper 直接从 name 的第 5、6+ 段解析（与 ou_list_ckpts.py 一致）。
- 自训：`tofu_1B_<Trainer>_forget10[_<run_tag>]`
  → task_name 里**没有超参串**，必须用 --method / --hyper 显式传，
    否则 hyper 会退化成 run_tag，无法与同名官方 ckpt 对拍。

用法：
    # 官方 ckpt：method/hyper 自动解析
    python scripts/ou_append_run.py \
      --name unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d0_g0.125_ep10 \
      --agg-json saves/eval/<name>/ou_aggregate.json \
      --repo-id open-unlearning/unlearn_tofu_..._SimNPO_lr1e-05_...

    # 自训：必须显式给 --method / --hyper
    python scripts/ou_append_run.py \
      --name tofu_1B_SimNPO_forget10_S1 \
      --agg-json /root/autodl-tmp/saves/unlearn/tofu_1B_SimNPO_forget10_S1/evals/ou_aggregate.json \
      --method SimNPO --hyper lr1e-05_b4.5_a1_d0_g0.125_ep10 --source selftrain
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

DIMS = ["Mem", "Priv", "Utility", "Agg"]
OFFICIAL_PREFIX = "unlearn_tofu_"
DEFAULT_RUNS = "results/ou_table3_runs.jsonl"


def parse_name(name):
    """从 name 猜 (method, hyper, source)。

    官方 ckpt：unlearn_tofu_<model>_<split>_<Method>_<超参串>
    自训：    tofu_1B_<Trainer>_forget10[_<run_tag>]（超参串不在 name 里，需显式传）
    """
    seg = name.split("_")
    if name.startswith(OFFICIAL_PREFIX) and len(seg) >= 6:
        return seg[4], "_".join(seg[5:]), "official"
    # tofu_1B_<Trainer>_forget10[_tag]
    if len(seg) >= 3 and seg[0] == "tofu":
        return seg[2], (seg[4] if len(seg) >= 5 else ""), "selftrain"
    return "?", name, "unknown"


def append_run(name, agg_json, runs_path, repo_id="", method=None, hyper=None,
               source=None, ckpt_path=""):
    """把一条四维结果写进 jsonl；同一 name 重跑时覆盖旧行。返回写入的 record。"""
    if not os.path.exists(agg_json):
        raise FileNotFoundError(f"四维聚合结果不存在：{agg_json}（先跑 ou_aggregate.py --json）")

    with open(agg_json) as f:
        agg = json.load(f)

    guess_method, guess_hyper, guess_source = parse_name(name)
    method = method or guess_method
    hyper = hyper or guess_hyper
    source = source or guess_source

    rec = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": name,
        "repo_id": repo_id,
        "method": method,
        "hyper": hyper,
        "source": source,
        "ckpt_path": ckpt_path,
    }
    for d in DIMS:
        rec[d] = agg[d]
    rec["components"] = agg.get("components")
    rec["params"] = agg.get("params")

    # 同一 name 只保留一行（重跑覆盖），其余原样保留
    kept = []
    if os.path.exists(runs_path):
        with open(runs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line)["name"] == name:
                        continue  # 覆盖旧行
                except (json.JSONDecodeError, KeyError):
                    pass  # 坏行保留，避免静默丢数据
                kept.append(line)
    kept.append(json.dumps(rec, ensure_ascii=False))

    # 原子写入：先写同目录临时文件再 rename，避免中断留下半截 jsonl
    os.makedirs(os.path.dirname(runs_path) or ".", exist_ok=True)
    d = os.path.dirname(runs_path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ou_runs_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(kept) + "\n")
        os.replace(tmp, runs_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return rec


def main():
    p = argparse.ArgumentParser(
        description="追加一条四维结果到 ou_table3_runs.jsonl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", required=True, help="ckpt 名或自训 task_name")
    p.add_argument("--agg-json", required=True, help="ou_aggregate.py --json 的输出")
    p.add_argument("--runs", default=DEFAULT_RUNS, help="jsonl 路径")
    p.add_argument("--repo-id", default="", help="官方 ckpt 的 HF repo id（自训留空）")
    p.add_argument("--method", default=None, help="方法名（自训必须显式传）")
    p.add_argument("--hyper", default=None, help="超参串（自训必须显式传，用于与官方 ckpt 对拍）")
    p.add_argument("--source", choices=["official", "selftrain", "unknown"], default=None,
                   help="来源（默认按 name 前缀自动判断）")
    p.add_argument("--ckpt-path", default="", help="自训权重路径（便于回溯）")
    args = p.parse_args()

    try:
        rec = append_run(
            name=args.name, agg_json=args.agg_json, runs_path=args.runs,
            repo_id=args.repo_id, method=args.method, hyper=args.hyper,
            source=args.source, ckpt_path=args.ckpt_path,
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"[{rec['source']}] {rec['method']:<10} {rec['hyper']:<40} "
          f"Mem={rec['Mem']:.4f} Priv={rec['Priv']:.4f} "
          f"Utility={rec['Utility']:.4f} Agg={rec['Agg']:.4f}")
    print(f"  -> {args.runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
