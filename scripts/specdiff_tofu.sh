#!/bin/bash
# SpecDiff TOFU-1B smoke and three-seed experiment.
#
#   bash scripts/specdiff_tofu.sh smoke
#   bash scripts/specdiff_tofu.sh seeds
#   bash scripts/specdiff_tofu.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
MODEL="Llama-3.2-1B-Instruct"
SPLIT="forget10"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
TAG_PREFIX="${SPECDIFF_TAG:+${SPECDIFF_TAG}_}"
LAM="${SPECDIFF_LAM:-1.0}"
BETA="${SPECDIFF_BETA:-0.1}"
KAPPA="${SPECDIFF_KAPPA:-0.3}"
TAU="${SPECDIFF_TAU:-0.02}"
COMMON_ARGS=(
  trainer.args.learning_rate=5e-5
  trainer.args.optim=adamw_torch
  trainer.args.do_eval=false
  trainer.args.eval_on_start=false
  trainer.args.eval_strategy=no
  trainer.method_args.draft_model_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
  trainer.method_args.lam="$LAM"
  trainer.method_args.beta="$BETA"
  trainer.method_args.kappa="$KAPPA"
  trainer.method_args.tau="$TAU"
  trainer.method_args.warmup_steps=1
  trainer.method_args.chunk_size=8192
)

run_one() {
  local tag="$1"
  shift
  local aggregate="$SAVES/unlearn/tofu_1B_SpecDiff_${SPLIT}_${tag}/evals/ou_aggregate.json"
  if [[ -f "$aggregate" ]]; then
    echo "[skip] completed SpecDiff $tag"
    return
  fi
  bash scripts/tofu_unlearn_one.sh SpecDiff "$SPLIT" "$MODEL" "$tag" \
    "${COMMON_ARGS[@]}" "$@"
}

run_smoke() {
  run_one smoke_s0 \
    trainer.args.seed=0 \
    trainer.args.logging_steps=1 \
    +trainer.args.max_steps=5 \
    trainer.args.num_train_epochs=1
  python - "$SAVES/unlearn/tofu_1B_SpecDiff_${SPLIT}_smoke_s0/evals/ou_aggregate.json" <<'PY'
import json
import math
import sys

result = json.load(open(sys.argv[1]))
required = ("Mem", "Priv", "Utility", "Agg", "SpecGap")
missing = [key for key in required if key not in result]
if missing:
    raise SystemExit(f"smoke output missing: {missing}")
values = [
    result["Mem"],
    result["Priv"],
    result["Utility"],
    result["Agg"],
    result["SpecGap"]["forget"]["mean"],
    result["SpecGap"]["retain"]["mean"],
]
if not all(math.isfinite(float(value)) for value in values):
    raise SystemExit("smoke output contains non-finite metrics")
print("[gate] SpecDiff smoke passed")
PY
}

run_seeds() {
  local seed
  for seed in 0 1 2; do
    run_one "${TAG_PREFIX}seed${seed}" \
      trainer.args.seed="$seed" \
      trainer.args.logging_steps=5 \
      trainer.args.num_train_epochs=10
  done
  python scripts/specdiff_table.py
}

case "$MODE" in
  smoke) run_smoke ;;
  seeds) run_seeds ;;
  all)
    run_smoke
    run_seeds
    ;;
  grid)
    python scripts/specdiff_grid.py
    ;;
  *) echo "usage: $0 [smoke|seeds|all|grid]"; exit 2 ;;
esac
