#!/bin/bash
# SimNPO 网格调参主入口（TOFU · Llama-3.2-1B-Instruct · forget10）
#
# 用法:
#   bash scripts/tofu_tune_simnpo.sh 0        # Stage 0 单卡轨迹冒烟（复现默认配置）
#   bash scripts/tofu_tune_simnpo.sh a1       # Stage A1 强度粗扫 beta×gamma（2 卡终点）
#   bash scripts/tofu_tune_simnpo.sh a2       # Stage A2 学习率（基于 a1 top-3）
#   bash scripts/tofu_tune_simnpo.sh b        # Stage B 停止点（top-3 单卡轨迹）
#   bash scripts/tofu_tune_simnpo.sh c        # Stage C 保效用（alpha/delta/KL）
#   bash scripts/tofu_tune_simnpo.sh final    # 最终确认（最优配置 2 卡完整重跑）
#
# 约定:
#   * 权重落盘后即删（轨迹模式仅删 model.safetensors，保留 checkpoint-*/evals）
#   * 已评测的 trial 自动跳过（断点续跑）
#   * 目录名编码超参，供 summarize_tune_trials.py 反解

set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
MODEL_HF="open-unlearning/tofu_${MODEL}_full"
RETAIN_LOG="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
TUNE_ROOT="/root/autodl-tmp/saves/tune/simnpo_forget10"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "$LOG_ROOT" "$TUNE_ROOT"

STAGE="${1:?stage required: 0|a1|a2|b|c|final}"
shift || true
EXTRA_OVERRIDES=("$@")

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"

# run_trial <mode 1gpu|2gpu> <beta> <gamma> <alpha> <delta> <lr> <epochs> [retain_loss_type]
run_trial() {
  local mode="$1" beta="$2" gamma="$3" alpha="$4" delta="$5" lr="$6" epochs="$7" rlt="${8:-NLL}"
  local task_name="tofu_${MODEL}_${FORGET_SPLIT}_SimNPO_b${beta}_g${gamma}_a${alpha}_d${delta}_lr${lr}_e${epochs}"
  local ckpt="$TUNE_ROOT/$task_name"
  mkdir -p "$ckpt"
  # 断点续跑：2 卡终点看 endpoint；单卡轨迹看 checkpoint-1/evals
  if [ "$mode" = "1gpu" ]; then
    if [ -f "$ckpt/checkpoint-1/evals/TOFU_SUMMARY.json" ]; then
      echo "[SKIP] $task_name (轨迹已存在)"; return 0
    fi
  else
    if [ -f "$ckpt/evals/TOFU_SUMMARY.json" ]; then
      echo "[SKIP] $task_name (终点已评测)"; return 0
    fi
  fi
  echo "===== TRIAL $task_name (mode=$mode, rlt=$rlt) ====="
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local logf="$LOG_ROOT/${task_name}_${ts}.log"

  if [ "$mode" = "1gpu" ]; then
    CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
      experiment=unlearn/tofu/default trainer=SimNPO task_name="$task_name" model="$MODEL" \
      forget_split="$FORGET_SPLIT" retain_split="$RETAIN_SPLIT" \
      model.model_args.pretrained_model_name_or_path="$MODEL_HF" \
      model.tokenizer_args.pretrained_model_name_or_path="$MODEL_HF" \
      retain_logs_path="$RETAIN_LOG" paths.output_dir="$ckpt" \
      trainer.args.per_device_train_batch_size=8 \
      trainer.args.gradient_accumulation_steps=4 \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.args.eval_strategy=epoch trainer.args.eval_on_start=False \
      trainer.args.num_train_epochs="$epochs" trainer.args.learning_rate="$lr" \
      trainer.method_args.beta="$beta" trainer.method_args.gamma="$gamma" \
      trainer.method_args.alpha="$alpha" trainer.method_args.delta="$delta" \
      trainer.method_args.retain_loss_type="$rlt" \
      "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$logf"
  else
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
      --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
      src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
      trainer=SimNPO task_name="$task_name" model="$MODEL" \
      forget_split="$FORGET_SPLIT" retain_split="$RETAIN_SPLIT" \
      model.model_args.pretrained_model_name_or_path="$MODEL_HF" \
      model.tokenizer_args.pretrained_model_name_or_path="$MODEL_HF" \
      retain_logs_path="$RETAIN_LOG" paths.output_dir="$ckpt" \
      trainer.args.per_device_train_batch_size=4 \
      trainer.args.gradient_accumulation_steps=4 \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.args.eval_strategy=no trainer.args.eval_on_start=False \
      trainer.args.num_train_epochs="$epochs" trainer.args.learning_rate="$lr" \
      trainer.method_args.beta="$beta" trainer.method_args.gamma="$gamma" \
      trainer.method_args.alpha="$alpha" trainer.method_args.delta="$delta" \
      trainer.method_args.retain_loss_type="$rlt" \
      "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "$logf"
  fi

  # 终点评测（统一产出 evals/TOFU_SUMMARY.json）
  CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split="$FORGET_SPLIT" holdout_split="$HOLDOUT_SPLIT" model="$MODEL" \
    task_name="$task_name" \
    model.model_args.pretrained_model_name_or_path="$ckpt" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL_HF" \
    paths.output_dir="$ckpt/evals" retain_logs_path="$RETAIN_LOG" 2>&1 | tee -a "$logf"

  # 清理权重：轨迹模式保留 checkpoint-*/evals；2 卡模式无 checkpoint 权重
  rm -f "$ckpt/model.safetensors"
  [ "$mode" = "2gpu" ] && rm -rf "$ckpt"/checkpoint-*
  echo "[DONE] $task_name"
}

# select_configs <count> <filter: any|traj>  -> 输出 TSV: beta gamma alpha delta lr epochs
select_configs() {
  python3 - "$TUNE_ROOT/trials.json" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); n = int(sys.argv[2]); filt = sys.argv[3]
trials = json.loads(p.read_text()) if p.exists() else []
if filt == "traj":
    trials = [t for t in trials if t.get("has_trajectory")]
cand = [t for t in trials if (t.get("metrics", {}).get("model_utility") or 0) >= 0.55]
cand.sort(key=lambda t: (t.get("metrics", {}).get("forget_quality") or -1), reverse=True)
for t in cand[:n]:
    print(f"{t['beta']}\t{t['gamma']}\t{t['alpha']}\t{t['delta']}\t{t['lr']}\t{t['epochs']}")
PY
}

case "$STAGE" in
  0)
    run_trial 1gpu 4.5 0.125 1.0 0.0 1e-5 10
    ;;
  a1)
    for beta in 0.5 1.0 2.5 4.5; do
      for gamma in 0.125 1.0; do
        run_trial 2gpu "$beta" "$gamma" 1.0 0.0 1e-5 10
      done
    done
    ;;
  a2)
    while IFS=$'\t' read -r beta gamma alpha delta lr epochs; do
      [ -z "${beta:-}" ] && continue
      run_trial 2gpu "$beta" "$gamma" "$alpha" "$delta" 2e-5 "$epochs"
      run_trial 2gpu "$beta" "$gamma" "$alpha" "$delta" 5e-5 "$epochs"
    done < <(select_configs 3 any)
    ;;
  b)
    while IFS=$'\t' read -r beta gamma alpha delta lr epochs; do
      [ -z "${beta:-}" ] && continue
      run_trial 1gpu "$beta" "$gamma" "$alpha" "$delta" "$lr" 10
    done < <(select_configs 3 any)
    ;;
  c)
    while IFS=$'\t' read -r beta gamma alpha delta lr epochs; do
      [ -z "${beta:-}" ] && continue
      best_beta="$beta"; best_gamma="$gamma"; best_lr="$lr"
    done < <(select_configs 1 any)
    [ -z "${best_beta:-}" ] && { echo "无可用 base 配置，先跑 a1/a2"; exit 1; }
    for a in 2.0 5.0; do
      for d in 0.0 0.05; do
        run_trial 2gpu "$best_beta" "$best_gamma" "$a" "$d" "$best_lr" 10
      done
    done
    run_trial 2gpu "$best_beta" "$best_gamma" "$alpha" 0.0 "$best_lr" 10 KL
    ;;
  final)
    while IFS=$'\t' read -r beta gamma alpha delta lr epochs; do
      [ -z "${beta:-}" ] && continue
      run_trial 2gpu "$beta" "$gamma" "$alpha" "$delta" "$lr" 10
    done < <(select_configs 1 any)
    ;;
  *)
    echo "未知 stage: $STAGE (应为 0|a1|a2|b|c|final)"; exit 1 ;;
esac

# 每次跑完汇总
python scripts/summarize_tune_trials.py
