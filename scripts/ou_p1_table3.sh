#!/bin/bash
# 阶段 A：官方 ckpt 免训评测（Table 3 主路径）。
# 等本机 TPO/训练释放 GPU → 冒烟 2 条 → SimNPO 48 → 命中后再 RMU 及其余。
#
# 用法：
#   bash scripts/ou_p1_table3.sh
# 可覆盖：GPU=0 LOG=/root/autodl-tmp/logs/ou_p1_table3.log
set -euo pipefail

ROOT=/usr/local/open-unlearning
cd "$ROOT"
source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

case "${HF_HOME:-}" in
  /root/autodl-tmp/*) ;;
  *) echo "[error] HF_HOME 必须在数据盘，当前=${HF_HOME:-unset}"; exit 1 ;;
esac

GPU="${GPU:-0,1}"
LOG="${LOG:-/root/autodl-tmp/logs/ou_p1_table3.log}"
mkdir -p "$(dirname "$LOG")"

gpu_busy () {
  # 只匹配真正的 python 评测/训练，避免 waiter 命令行里的 src/eval.py 误伤
  pgrep -f 'python(3)? .*(src/eval\.py|src/train\.py)' >/dev/null 2>&1 && return 0
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  [[ "${mem:-0}" -ge 2000 ]]
}

echo "=== [$(date +%F\ %T)] ou_p1_table3 等待 GPU（TPO 评测结束）==="
for i in $(seq 1 240); do
  if ! gpu_busy; then
    echo "GPU free at $(date +%T) iter=$i"
    break
  fi
  echo "wait $i $(date +%T) still busy"
  sleep 30
done
if gpu_busy; then
  echo "[error] 等待超时，GPU 仍忙"
  exit 1
fi

if grep -q '"source": "official"' "$ROOT/results/ou_table3_runs.jsonl" 2>/dev/null; then
  echo "=== 跳过冒烟（jsonl 已有 official 行）==="
else
  echo "=== 冒烟 2 条 ==="
  bash scripts/ou_eval_batch.sh --limit 2 --gpu "$GPU"
fi
python scripts/ou_table.py --source official

echo "=== SimNPO 48 ==="
bash scripts/ou_eval_batch.sh --only SimNPO --gpu "$GPU"
python scripts/ou_table.py --source official

read -r N_HIT N_SIM < <(python scripts/ou_official_hit_count.py SimNPO 0.02)
echo "SimNPO official hits=${N_HIT:-0} / ${N_SIM:-0}"

if [[ "${N_HIT:-0}" -lt 1 ]]; then
  echo "[stop] SimNPO missed paper target within 0.02; skip RMU and retrain."
  python scripts/ou_table.py --source official
  exit 0
fi

echo "=== RMU 54 ==="
bash scripts/ou_eval_batch.sh --only RMU --gpu "$GPU"
echo "=== GradDiff ==="
bash scripts/ou_eval_batch.sh --only GradDiff --gpu "$GPU"
echo "=== 清单剩余 ==="
bash scripts/ou_eval_batch.sh --gpu "$GPU"
python scripts/ou_table.py --source official
echo "=== [$(date +%F\ %T)] P1 完成 ==="
