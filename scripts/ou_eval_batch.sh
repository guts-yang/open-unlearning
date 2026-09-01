#!/bin/bash
# P1 批量驱动：按 results/ckpt_list.json 串行跑 scripts/ou_eval_one.sh。
#
# 用法：
#   bash scripts/ou_eval_batch.sh                        # 跑清单里全部 pending
#   bash scripts/ou_eval_batch.sh --only SimNPO          # 只跑某个方法
#   bash scripts/ou_eval_batch.sh --limit 2              # 先试跑 2 条（强烈建议先做）
#   bash scripts/ou_eval_batch.sh --only RMU --limit 4
#   bash scripts/ou_eval_batch.sh --force                # 忽略 skip-done，重跑
#
# 行为：
#   - 默认 --skip-done：results/ou_table3_runs.jsonl 里已有该 ckpt 就跳过（断点续跑）
#   - 单条失败不中断整批，失败名写入 results/ou_table3_failures.log
#   - 每条打印进度与数据盘剩余空间
set -euo pipefail

ROOT=/usr/local/open-unlearning
LIST="${LIST:-$ROOT/results/ckpt_list.json}"
RUNS="${RUNS:-$ROOT/results/ou_table3_runs.jsonl}"
GPU="${GPU:-0}"

ONLY=""
LIMIT=""
SKIP_DONE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)  ONLY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) SKIP_DONE=0; shift ;;
    --list)  LIST="$2"; shift 2 ;;
    --gpu)   GPU="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

[[ -f "$LIST" ]] || { echo "[error] 清单不存在：$LIST（先跑 scripts/ou_list_ckpts.py）"; exit 1; }

mapfile -t ENTRIES < <(python - "$LIST" "$ONLY" "$LIMIT" "$SKIP_DONE" "$RUNS" <<'PY'
import json, os, sys

list_path, only, limit, skip_done, runs_path = sys.argv[1:6]
skip_done = int(skip_done)
with open(list_path) as f:
    ckpts = json.load(f)["ckpts"]

done = set()
if skip_done and os.path.exists(runs_path):
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["name"])

out = []
for c in ckpts:
    if only and c["method"] != only:
        continue
    if skip_done and c["name"] in done:
        continue
    if c.get("status") == "failed":
        continue
    out.append(c["name"])

if limit:
    out = out[: int(limit)]
print("\n".join(out))
PY
)

TOTAL=${#ENTRIES[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "[info] 没有待跑的 ckpt（清单已跑完或 --only 过滤后为空）"
  exit 0
fi

echo "=== 待跑 $TOTAL 条（GPU $GPU，清单 $LIST）==="
i=0
for NAME in "${ENTRIES[@]}"; do
  i=$((i + 1))
  echo "----- [$i/$TOTAL] $NAME -----"
  if bash "$ROOT/scripts/ou_eval_one.sh" "$NAME" "$GPU"; then
    echo "[ok] $NAME"
  else
    echo "[fail] $NAME（已记入 results/ou_table3_failures.log，继续下一条）"
  fi
done

echo "=== 批量结束：$TOTAL 条。失败清单（若有）==="
[[ -f "$ROOT/results/ou_table3_failures.log" ]] && tail -20 "$ROOT/results/ou_table3_failures.log" || true
echo "四维汇总表：python scripts/ou_table.py"
