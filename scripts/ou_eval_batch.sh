#!/bin/bash
# P1 批量驱动：按 results/ckpt_list.json 跑 scripts/ou_eval_one.sh。
# 默认 GPU_IDS=0,1：两张卡各评一条，一条结束立刻补下一条。
#
# 用法：
#   bash scripts/ou_eval_batch.sh                        # 跑清单里全部 pending
#   GPU=0,1 bash scripts/ou_eval_batch.sh --only SimNPO
#   bash scripts/ou_eval_batch.sh --gpu 0 --only SimNPO  # 强制单卡
#   bash scripts/ou_eval_batch.sh --limit 2
#   bash scripts/ou_eval_batch.sh --force
set -euo pipefail

ROOT=/usr/local/open-unlearning
LIST="${LIST:-$ROOT/results/ckpt_list.json}"
RUNS="${RUNS:-$ROOT/results/ou_table3_runs.jsonl}"
GPU="${GPU:-0,1}"
# 评测同时预取后续条数（每条约 2.5GB，4 条约 10GB，只放数据盘）
PREFETCH="${PREFETCH:-4}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"

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

case "$SAVES" in
  /root/autodl-tmp/*) ;;
  *) echo "[error] SAVES 必须在数据盘，当前=$SAVES"; exit 1 ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU"
echo "=== 待跑 $TOTAL 条（GPU ${GPUS[*]}，prefetch=$PREFETCH，ckpt=$SAVES/hf_ckpts）==="

# 后台预取 [from, from+PREFETCH)，已齐的立刻返回
prefetch_from () {
  local from="$1"
  local k=0
  local n
  while [[ $k -lt $PREFETCH ]]; do
    n=$((from + k))
    if [[ $n -ge $TOTAL ]]; then
      break
    fi
    bash "$ROOT/scripts/ou_download_ckpt.sh" "${ENTRIES[$n]}" &
    k=$((k + 1))
  done
}

# 先把第一批拉到数据盘，GPU 再开工
echo "[prefetch] 启动前预取 ${PREFETCH} 条"
prefetch_from 0
wait || true

i=0
while [[ $i -lt $TOTAL ]]; do
  # 评这一批的同时，预取再往后 PREFETCH 条
  prefetch_from $((i + ${#GPUS[@]}))
  pids=()
  names=()
  gused=()
  for g in "${GPUS[@]}"; do
    if [[ $i -ge $TOTAL ]]; then
      break
    fi
    NAME="${ENTRIES[$i]}"
    i=$((i + 1))
    echo "----- [$i/$TOTAL] $NAME  GPU=$g -----"
    bash "$ROOT/scripts/ou_eval_one.sh" "$NAME" "$g" &
    pids+=("$!")
    names+=("$NAME")
    gused+=("$g")
  done
  j=0
  while [[ $j -lt ${#pids[@]} ]]; do
    rc=0
    wait "${pids[$j]}" || rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "[ok] ${names[$j]} (GPU ${gused[$j]})"
    else
      echo "[fail] ${names[$j]}（已记入 results/ou_table3_failures.log，继续）"
    fi
    j=$((j + 1))
  done
done
# 收掉可能还在跑的预取
wait || true

echo "=== 批量结束：$TOTAL 条。失败清单（若有）==="
[[ -f "$ROOT/results/ou_table3_failures.log" ]] && tail -20 "$ROOT/results/ou_table3_failures.log" || true
echo "四维汇总表：python scripts/ou_table.py --source official"
