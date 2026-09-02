#!/bin/bash
# 对已有 TOFU_SUMMARY.json 的评测目录重跑四维聚合并写入 jsonl（不重评、不下载）。
# 无参：扫描 $SAVES/eval/unlearn_* 以及自训 G1/GA1。
set -euo pipefail

ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV=/root/autodl-tmp/envs/unlearning
RUNS="$ROOT/results/ou_table3_runs.jsonl"

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"

LOCAL_RETAIN="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_retain90_local"
LOCAL_INIT="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10"
RETAIN_SUMMARY="$LOCAL_RETAIN/TOFU_SUMMARY.json"
INIT_SUMMARY="$LOCAL_INIT/TOFU_SUMMARY.json"

cd "$ROOT"

backfill_one() {
  local eval_dir="$1" name
  name="$(basename "$eval_dir")"
  # 自训产物在 saves/unlearn/<task>/evals/
  if [[ "$name" == evals ]]; then
    name="$(basename "$(dirname "$eval_dir")")"
  fi
  if [[ ! -f "$eval_dir/TOFU_SUMMARY.json" || ! -f "$eval_dir/TOFU_EVAL.json" ]]; then
    echo "[skip] $name 缺 SUMMARY/EVAL"
    return 0
  fi
  if ! python -c "import json,sys; s=json.load(open(sys.argv[1])); raise SystemExit(0 if 'extraction_strength' in s else 1)" \
        "$eval_dir/TOFU_SUMMARY.json"; then
    echo "[skip] $name SUMMARY 指标未齐（评测进行中）"
    return 0
  fi
  python scripts/ou_aggregate.py "$eval_dir/TOFU_SUMMARY.json" \
    --init-summary "$INIT_SUMMARY" \
    --retain-summary "$RETAIN_SUMMARY" \
    --json "$eval_dir/ou_aggregate.json"
  case "$name" in
    tofu_1B_GradDiff_forget10_G1)
      python scripts/ou_append_run.py --name "$name" \
        --agg-json "$eval_dir/ou_aggregate.json" --runs "$RUNS" \
        --method GradDiff --hyper lr1e-05_alpha5_epoch10 --source selftrain \
        --ckpt-path "$SAVES/unlearn/$name"
      ;;
    tofu_1B_GradAscent_forget10_GA1)
      python scripts/ou_append_run.py --name "$name" \
        --agg-json "$eval_dir/ou_aggregate.json" --runs "$RUNS" \
        --method GradAscent --hyper default_lr1e-05_ep10 --source selftrain \
        --ckpt-path "$SAVES/unlearn/$name"
      ;;
    unlearn_*)
      python scripts/ou_append_run.py --name "$name" \
        --agg-json "$eval_dir/ou_aggregate.json" --runs "$RUNS" \
        --repo-id "open-unlearning/$name" --source official
      ;;
    *)
      echo "[skip] 未知 name 格式，只写了 ou_aggregate.json：$name"
      ;;
  esac
}

if [[ $# -gt 0 ]]; then
  for d in "$@"; do backfill_one "$d"; done
else
  shopt -s nullglob
  for d in "$SAVES"/eval/unlearn_tofu_*; do
    backfill_one "$d"
  done
  for tag in tofu_1B_GradDiff_forget10_G1 tofu_1B_GradAscent_forget10_GA1; do
    if [[ -f "$SAVES/unlearn/$tag/evals/TOFU_SUMMARY.json" ]]; then
      backfill_one "$SAVES/unlearn/$tag/evals"
    fi
  done
fi
