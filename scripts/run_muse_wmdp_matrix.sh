#!/bin/bash
# MUSE / WMDP 复现矩阵。默认 --full 两卡并行、每卡一组不同实验。
#
#   bash scripts/run_muse_wmdp_matrix.sh --full          # 2 worker
#   bash scripts/run_muse_wmdp_matrix.sh --full --serial # 单卡排队
#   PARALLEL=1 bash scripts/run_muse_wmdp_matrix.sh --p0
#
# 失败不中断。跳过 jsonl 里已 ok 的正式组，以及进程里已在跑的组。
# 日志：$DATA/logs/muse_wmdp_matrix.log ；每卡 $DATA/logs/muse_wmdp_gpu{N}.log
set -uo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/muse_wmdp_matrix.log"
LOCK_DIR="$LOG_DIR/muse_wmdp_claim"
mkdir -p "$LOCK_DIR"
CLAIM_LOCK="$LOCK_DIR/queue.lock"

MODE=p0
PARALLEL="${PARALLEL:-2}"
for arg in "$@"; do
  case "$arg" in
    --full) MODE=full ;;
    --p0) MODE=p0 ;;
    --serial) PARALLEL=1 ;;
  esac
done

P0_MUSE=(GradAscent GradDiff NPO SimNPO RMU)
MORE=(DPO UNDIAL CEU WGA SatImp PDU)

# 行格式：kind trainer [split]
JOBS=()
for t in "${P0_MUSE[@]}"; do JOBS+=("muse $t News"); done
JOBS+=("wmdp RMU")
if [[ "$MODE" == "full" ]]; then
  for t in "${P0_MUSE[@]}"; do JOBS+=("muse $t Books"); done
  for split in News Books; do
    for t in "${MORE[@]}"; do JOBS+=("muse $t $split"); done
  done
  for t in GradAscent GradDiff NPO SimNPO UNDIAL CEU WGA SatImp; do
    JOBS+=("wmdp $t")
  done
fi

job_id () {
  local kind="$1" trainer="$2" split="${3:-cyber}"
  if [[ "$kind" == muse ]]; then
    echo "muse_${split}_${trainer}"
  else
    echo "wmdp_${split:-cyber}_${trainer}"
  fi
}

task_name_of () {
  local kind="$1" trainer="$2" split="${3:-cyber}"
  if [[ "$kind" == muse ]]; then
    echo "muse_Llama-2-7b-hf_${split}_${trainer}"
  else
    echo "wmdp_zephyr_${split:-cyber}_${trainer}"
  fi
}

jsonl_ok () {
  local task="$1"
  python3 - "$task" <<'PY'
import json, sys
from pathlib import Path
task = sys.argv[1]
p = Path("/usr/local/open-unlearning/results/muse_wmdp_runs.jsonl")
if not p.exists():
    raise SystemExit(1)
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("task_name") != task:
        continue
    if r.get("status") == "ok" and "skip_eval" not in (r.get("note") or ""):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

job_running () {
  local kind="$1" trainer="$2" split="${3:-}"
  local task
  task="$(task_name_of "$kind" "$trainer" "$split")"
  if pgrep -af "src/train.py" | grep -Eq "task_name=${task}( |$)"; then
    return 0
  fi
  if pgrep -af "src/eval.py" | grep -Eq "task_name=${task}( |$)"; then
    return 0
  fi
  if [[ "$kind" == muse ]]; then
    pgrep -af "scripts/muse_unlearn_one.sh ${trainer} ${split}" | grep -vq pgrep && return 0
  else
    pgrep -af "scripts/wmdp_unlearn_one.sh ${trainer}" | grep -vq pgrep && return 0
  fi
  return 1
}

gpu_busy () {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'
}

cgroup_free_gib () {
  python3 - <<'PY'
lim=int(open("/sys/fs/cgroup/memory/memory.limit_in_bytes").read())
used=int(open("/sys/fs/cgroup/memory/memory.usage_in_bytes").read())
print(int((lim-used)/1024/1024/1024))
PY
}

# 原子领取下一组：不重复、不领正在跑的、不领已 ok 的。
claim_job () {
  local gpu="$1"
  local line kind trainer split id task
  exec 9>"$CLAIM_LOCK"
  flock 9
  for line in "${JOBS[@]}"; do
    # shellcheck disable=SC2086
    set -- $line
    kind="$1"; trainer="$2"; split="${3:-cyber}"
    id="$(job_id "$kind" "$trainer" "$split")"
    task="$(task_name_of "$kind" "$trainer" "$split")"
    if [[ -f "$LOCK_DIR/${id}.claimed" || -f "$LOCK_DIR/${id}.done_fail" ]]; then
      continue
    fi
    if jsonl_ok "$task"; then
      continue
    fi
    if job_running "$kind" "$trainer" "$split"; then
      continue
    fi
    printf '%s\n' "$gpu $(date '+%F %T')" > "$LOCK_DIR/${id}.claimed"
    echo "$kind $trainer $split"
    flock -u 9
    exec 9>&-
    return 0
  done
  flock -u 9
  exec 9>&-
  return 1
}

run_claimed () {
  local gpu="$1" kind="$2" trainer="$3" split="${4:-}"
  local id
  id="$(job_id "$kind" "$trainer" "$split")"
  echo "======== $(date '+%F %T') gpu$gpu $kind $trainer ${split} ========" | tee -a "$MAIN_LOG"
  # 单卡一份 7B；禁止 NUM_PROCESSES=2 把同一实验拆到两卡。
  export NUM_PROCESSES=1
  export GPU_IDS="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export EVAL_GPU="$gpu"
  local ok=0
  if [[ "$kind" == muse ]]; then
    if bash scripts/muse_unlearn_one.sh "$trainer" "$split"; then ok=1; fi
  else
    if bash scripts/wmdp_unlearn_one.sh "$trainer"; then ok=1; fi
  fi
  if [[ "$ok" == 1 ]]; then
    echo "[OK] gpu$gpu $kind $trainer ${split}" | tee -a "$MAIN_LOG"
  else
    echo "[FAIL] gpu$gpu $kind $trainer ${split}" | tee -a "$MAIN_LOG"
    echo "$(date '+%F %T') gpu$gpu" > "$LOCK_DIR/${id}.done_fail"
  fi
  rm -f "$LOCK_DIR/${id}.claimed"
}

worker () {
  local gpu="$1"
  local wlog="$LOG_DIR/muse_wmdp_gpu${gpu}.log"
  echo "[info] worker gpu$gpu start $(date '+%F %T')" | tee -a "$MAIN_LOG" "$wlog"
  while true; do
    if gpu_busy "$gpu"; then
      sleep 30
      continue
    fi
    local free
    free="$(cgroup_free_gib || echo 0)"
    if [[ "${free:-0}" -lt 8 ]]; then
      echo "[info] gpu$gpu wait cgroup free=${free}GiB <8 $(date '+%F %T')" | tee -a "$wlog"
      sleep 45
      continue
    fi
    local claimed
    if ! claimed="$(claim_job "$gpu")"; then
      echo "[info] worker gpu$gpu 队列空 $(date '+%F %T')" | tee -a "$MAIN_LOG" "$wlog"
      break
    fi
    # shellcheck disable=SC2086
    run_claimed "$gpu" $claimed >>"$wlog" 2>&1
  done
}

rm -f "$LOCK_DIR"/*.claimed
echo "[info] mode=$MODE parallel=$PARALLEL jobs=${#JOBS[@]} $(date '+%F %T')" | tee -a "$MAIN_LOG"

if [[ "$PARALLEL" -le 1 ]]; then
  worker 0
else
  worker 0 &
  w0=$!
  sleep 20
  worker 1 &
  w1=$!
  wait "$w0" "$w1" || true
fi

echo "[info] 矩阵结束 mode=$MODE $(date '+%F %T')" | tee -a "$MAIN_LOG"
