#!/bin/bash
# SpecDiff transfer to MUSE (Llama-2-7B News/Books) and WMDP-cyber (Zephyr-7B).
# Does not use AltPO / DPO: those corpora have no y_a. Draft = the training start model.
#
#   bash scripts/specdiff_muse_wmdp.sh preflight
#   bash scripts/specdiff_muse_wmdp.sh news      # after TOFU GPU is free
#   bash scripts/specdiff_muse_wmdp.sh books
#   bash scripts/specdiff_muse_wmdp.sh wmdp
#   bash scripts/specdiff_muse_wmdp.sh all
#
# Locked method args match TOFU κ=0.3: λ=1, β=0.1, τ=0.02, warmup=1 GradDiff.
# Learning rate follows each bench's official recipe (MUSE 1e-5 / 10 ep; WMDP 5e-5 / 80 steps).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE
unset CUDA_VISIBLE_DEVICES

MODE="${1:-preflight}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/huggingface/hub}"

SPECDIFF_METHOD=(
  trainer.method_args.lam=1.0
  trainer.method_args.beta=0.1
  trainer.method_args.kappa=0.3
  trainer.method_args.tau=0.02
  trainer.method_args.warmup_steps=1
  trainer.method_args.warmup_kind=graddiff
  trainer.method_args.dpo_weight=0.0
  trainer.method_args.chunk_size=8192
)

hub_ok() {
  local spec="$1"
  local dir="$CACHE/models--${spec//\//--}"
  [[ -d "$dir/snapshots" ]] && ls -d "$dir"/snapshots/* >/dev/null 2>&1
}

ds_ok() {
  local spec="$1"
  local dir="$CACHE/datasets--${spec//\//--}"
  [[ -d "$dir/snapshots" ]] && ls -d "$dir"/snapshots/* >/dev/null 2>&1
}

preflight() {
  local fail=0
  echo "[preflight] SpecDiff MUSE/WMDP"
  echo "  News ckpt   $SAVES/unlearn/muse_Llama-2-7b-hf_News_SpecDiff_k0.3_lr1e-5_ep10"
  echo "  Books ckpt  $SAVES/unlearn/muse_Llama-2-7b-hf_Books_SpecDiff_k0.3_lr1e-5_ep10"
  echo "  WMDP ckpt   $SAVES/unlearn/wmdp_zephyr-7b-beta_cyber_SpecDiff_k0.3_lr5e-5_ms80"
  echo "  table       $ROOT/results/muse_wmdp.md  (jsonl: results/muse_wmdp_runs.jsonl)"
  for spec in muse-bench/MUSE-News_target muse-bench/MUSE-Books_target HuggingFaceH4/zephyr-7b-beta NousResearch/Llama-2-7b-hf; do
    if hub_ok "$spec"; then
      echo "  [ok] model $spec"
    else
      echo "  [missing] model $spec"; fail=1
    fi
  done
  for spec in muse-bench/MUSE-News muse-bench/MUSE-Books; do
    if ds_ok "$spec"; then
      echo "  [ok] dataset $spec"
    else
      echo "  [missing] dataset $spec"; fail=1
    fi
  done
  for f in \
    "$ROOT/data/wmdp/wmdp-corpora/cyber-forget-corpus.jsonl" \
    "$ROOT/data/wmdp/wmdp-corpora/cyber-retain-corpus.jsonl"
  do
    if [[ -f "$f" ]]; then echo "  [ok] $f"; else echo "  [missing] $f"; fail=1; fi
  done
  for f in \
    "$SAVES/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json" \
    "$SAVES/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json"
  do
    if [[ -f "$f" ]]; then echo "  [ok] retain logs $f"
    elif [[ -f "${f/#$SAVES/$ROOT/saves}" ]]; then echo "  [ok] retain logs (repo) ${f/#$SAVES/saves}"
    else echo "  [warn] retain logs $f (muse_unlearn_one.sh will fall back)"
    fi
  done
  echo "  note: 7B SpecDiff loads a frozen draft (second 7B). Needs 2×80GB; KEEP_CKPT default 0 after eval."
  echo "  note: no AltPO on MUSE/WMDP (no alternate answers)."
  return "$fail"
}

run_if_needed() {
  local summary="$1"
  shift
  if [[ -f "$summary" ]]; then
    echo "[skip] $summary"
    return 0
  fi
  "$@"
}

run_news() {
  local tag="k0.3_lr1e-5_ep10"
  local ckpt="$SAVES/unlearn/muse_Llama-2-7b-hf_News_SpecDiff_${tag}"
  run_if_needed "$ckpt/evals/MUSE_SUMMARY.json" \
    env KEEP_CKPT="${KEEP_CKPT:-0}" RUN_TAG="$tag" \
    bash scripts/muse_unlearn_one.sh SpecDiff News \
      trainer.method_args.draft_model_path=muse-bench/MUSE-News_target \
      trainer.args.learning_rate=1e-5 \
      trainer.args.num_train_epochs=10 \
      "${SPECDIFF_METHOD[@]}"
}

run_books() {
  local tag="k0.3_lr1e-5_ep10"
  local ckpt="$SAVES/unlearn/muse_Llama-2-7b-hf_Books_SpecDiff_${tag}"
  run_if_needed "$ckpt/evals/MUSE_SUMMARY.json" \
    env KEEP_CKPT="${KEEP_CKPT:-0}" RUN_TAG="$tag" \
    bash scripts/muse_unlearn_one.sh SpecDiff Books \
      trainer.method_args.draft_model_path=muse-bench/MUSE-Books_target \
      trainer.args.learning_rate=1e-5 \
      trainer.args.num_train_epochs=10 \
      "${SPECDIFF_METHOD[@]}"
}

run_wmdp() {
  local tag="k0.3_lr5e-5_ms80"
  local ckpt="$SAVES/unlearn/wmdp_zephyr-7b-beta_cyber_SpecDiff_${tag}"
  if [[ -f "$ckpt/evals/LMEval_SUMMARY.json" || -f "$ckpt/evals/LM_EVAL_SUMMARY.json" ]]; then
    echo "[skip] $ckpt evals"
    return 0
  fi
  KEEP_CKPT="${KEEP_CKPT:-0}" RUN_TAG="$tag" \
    bash scripts/wmdp_unlearn_one.sh SpecDiff cyber \
      trainer.method_args.draft_model_path=HuggingFaceH4/zephyr-7b-beta \
      trainer.args.learning_rate=5e-5 \
      trainer.args.max_steps=80 \
      "${SPECDIFF_METHOD[@]}"
}

run_news_warmup() {
  local tag="warmup_only"
  local ckpt="$SAVES/unlearn/muse_Llama-2-7b-hf_News_SpecDiff_${tag}"
  run_if_needed "$ckpt/evals/MUSE_SUMMARY.json" \
    env KEEP_CKPT="${KEEP_CKPT:-0}" RUN_TAG="$tag" \
    bash scripts/muse_unlearn_one.sh SpecDiff News \
      trainer.method_args.draft_model_path=muse-bench/MUSE-News_target \
      trainer.args.learning_rate=1e-5 \
      trainer.args.num_train_epochs=1 \
      +trainer.args.max_steps=1 \
      "${SPECDIFF_METHOD[@]}"
}

case "$MODE" in
  preflight) preflight ;;
  news) preflight; run_news ;;
  books) preflight; run_books ;;
  wmdp) preflight; run_wmdp ;;
  warmup-news) preflight; run_news_warmup ;;
  all)
    preflight
    run_news_warmup
    run_news
    run_books
    run_wmdp
    ;;
  *)
    echo "usage: $0 [preflight|warmup-news|news|books|wmdp|all]"
    exit 2
    ;;
esac
