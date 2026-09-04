#!/bin/bash
# Sequential 2-GPU MUSE/WMDP matrix. Default = P0 (News 5 methods + WMDP RMU).
#   bash scripts/run_muse_wmdp_matrix.sh
#   bash scripts/run_muse_wmdp_matrix.sh --full
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE

FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --serial) ;; # sequential is the only mode here (one 2-GPU job at a time)
    *) echo "unknown arg $arg"; exit 1 ;;
  esac
done

LOG=/root/autodl-tmp/logs/muse_wmdp_matrix.log
mkdir -p /root/autodl-tmp/logs

run_one() {
  local cmd="$1"
  echo "======== $(date '+%F %T') START $cmd ========" | tee -a "$LOG"
  if eval "$cmd"; then
    echo "======== $(date '+%F %T') OK $cmd ========" | tee -a "$LOG"
  else
    echo "======== $(date '+%F %T') FAIL $cmd (continue) ========" | tee -a "$LOG"
  fi
}

# P0: official repro methods, News + WMDP RMU
P0_MUSE_NEWS=(GradAscent GradDiff NPO SimNPO RMU)
for t in "${P0_MUSE_NEWS[@]}"; do
  run_one "bash scripts/muse_unlearn_one.sh $t News"
done
run_one "bash scripts/wmdp_unlearn_one.sh RMU"

if [[ "$FULL" -eq 1 ]]; then
  for t in GradAscent GradDiff NPO SimNPO RMU; do
    run_one "bash scripts/muse_unlearn_one.sh $t Books"
  done
  for t in DPO UNDIAL CEU WGA SatImp PDU; do
    run_one "bash scripts/muse_unlearn_one.sh $t News"
    run_one "bash scripts/muse_unlearn_one.sh $t Books"
  done
  for t in GradAscent GradDiff NPO SimNPO UNDIAL CEU WGA SatImp; do
    run_one "bash scripts/wmdp_unlearn_one.sh $t"
  done
fi

echo "======== $(date '+%F %T') MATRIX DONE ========" | tee -a "$LOG"
