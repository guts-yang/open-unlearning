#!/bin/bash
# Train one TOFU unlearning method, then eval, then refresh results/tofu_<model>.md
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

trainer="${1:?usage: tofu_unlearn_one.sh TRAINER [forget_split] [model]}"
forget_split="${2:-forget10}"
model="${3:-Llama-3.2-1B-Instruct}"

case "$forget_split" in
  forget10) retain_split=retain90; holdout_split=holdout10 ;;
  forget05) retain_split=retain95; holdout_split=holdout05 ;;
  forget01) retain_split=retain99; holdout_split=holdout01 ;;
  *) echo "unknown forget_split $forget_split"; exit 1 ;;
esac

task_name="tofu_1B_${trainer}_${forget_split}"
ckpt="/root/autodl-tmp/saves/unlearn/${task_name}"
log_dir="/root/autodl-tmp/logs"
mkdir -p "$log_dir" "$ckpt"
model_hf="open-unlearning/tofu_${model}_full"

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$forget_split"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer="$trainer" \
  task_name="$task_name" \
  model="$model" \
  forget_split="$forget_split" retain_split="$retain_split" \
  model.model_args.pretrained_model_name_or_path="$model_hf" \
  model.tokenizer_args.pretrained_model_name_or_path="$model_hf" \
  retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json" \
  paths.output_dir="$ckpt" \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true

CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split="$forget_split" holdout_split="$holdout_split" \
  model="$model" \
  task_name="$task_name" \
  model.model_args.pretrained_model_name_or_path="$ckpt" \
  model.tokenizer_args.pretrained_model_name_or_path="$ckpt" \
  paths.output_dir="$ckpt/evals" \
  retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

python scripts/record_tofu_result.py \
  --summary "$ckpt/evals/TOFU_SUMMARY.json" \
  --method "$trainer" \
  --forget-split "$forget_split" \
  --model "$model" \
  --ckpt "$ckpt"
