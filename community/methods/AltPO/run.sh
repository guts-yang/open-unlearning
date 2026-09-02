#!/bin/bash
# AltPO 单组：对齐官方评测最好的超参串 lr2e-05_beta0.05_alpha1_epoch10
# （Agg 0.5826，见 results/ou_table3.md 第三节）。不再扫 lr/beta/alpha 网格。
#
# 用法：
#   bash community/methods/AltPO/run.sh                 # forget10 + 上述超参 + 标签 A1
#   bash community/methods/AltPO/run.sh forget10 A1
#
# 可覆盖：
#   GPU_IDS=0,1  NUM_PROCESSES=2  PER_DEVICE_BS=4  GRAD_ACCUM=4
#   LR=2e-5  BETA=0.05  ALPHA=1  EPOCHS=10
#   EXTRA_HYDRA  默认已含 V100 必需的 sdpa + adamw_torch
#
# 前置：community/methods/AltPO/data/.../alt5_seed_0.json
#   缺失时：cd community/methods/AltPO && python generate.py
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

FORGET_SPLIT="${1:-forget10}"
RUN_TAG="${2:-A1}"

case "$FORGET_SPLIT" in
  forget10) RETAIN_SPLIT=retain90; HOLDOUT_SPLIT=holdout10 ;;
  forget05) RETAIN_SPLIT=retain95; HOLDOUT_SPLIT=holdout05 ;;
  forget01) RETAIN_SPLIT=retain99; HOLDOUT_SPLIT=holdout01 ;;
  *) echo "unknown forget_split: $FORGET_SPLIT"; exit 1 ;;
esac

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
# 官方文件名 lr2e-05；Hydra 写 2e-5 同一数值
LR="${LR:-2e-5}"
BETA="${BETA:-0.05}"
ALPHA="${ALPHA:-1}"
EPOCHS="${EPOCHS:-10}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
# 与官方 ckpt 超参串逐字对齐（对拍用）
HYPER="${HYPER:-lr2e-05_beta0.05_alpha1_epoch10}"

EXTRA_HYDRA="${EXTRA_HYDRA:-model.model_args.attn_implementation=sdpa trainer.args.optim=adamw_torch}"
EXTRA_HYDRA_ARGS=()
# shellcheck disable=SC2206
read -r -a EXTRA_HYDRA_ARGS <<< "${EXTRA_HYDRA}"
EXTRA_TRAIN_HYDRA=()
EXTRA_EVAL_HYDRA=()
for _h in "${EXTRA_HYDRA_ARGS[@]}"; do
  if [[ "$_h" == trainer.* ]]; then
    EXTRA_TRAIN_HYDRA+=("$_h")
  else
    EXTRA_EVAL_HYDRA+=("$_h")
  fi
done

TASK_NAME="tofu_1B_AltPO_${FORGET_SPLIT}${RUN_TAG:+_${RUN_TAG}}"
CKPT="$SAVES/unlearn/$TASK_NAME"
MODEL_HF="open-unlearning/tofu_${MODEL}_full"
DATA_FILE="community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/${FORGET_SPLIT}/alt5_seed_0.json"

case "${HF_HOME:-}" in
  /root/autodl-tmp/*) ;;
  *) echo "[error] HF_HOME 不在数据盘：${HF_HOME:-unset}"; exit 1 ;;
esac
case "$SAVES" in
  /root/autodl-tmp/*) ;;
  *) echo "[error] SAVES 不在数据盘：$SAVES"; exit 1 ;;
esac

mkdir -p "$LOG_DIR" "$CKPT"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "[error] AltPO 交替回答数据缺失：$DATA_FILE"
  echo "        生成：source /root/autodl-tmp/env_hf.sh && cd $ROOT/community/methods/AltPO && python generate.py"
  exit 1
fi

LOCAL_RETAIN="$SAVES/eval/tofu_${MODEL}_${RETAIN_SPLIT}_local/TOFU_EVAL.json"
if [[ -f "$LOCAL_RETAIN" ]]; then
  RETAIN_LOGS="$LOCAL_RETAIN"
else
  RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
  echo "[warn] 未找到 P0-3 本地 retain 日志，回退 $RETAIN_LOGS"
fi

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"

echo "=== [$(date +%F\ %T)] AltPO: split=$FORGET_SPLIT hyper=$HYPER ==="
echo "    task=$TASK_NAME  lr=$LR beta=$BETA alpha=$ALPHA ep=$EPOCHS"
echo "    batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}  GPUs=$GPU_IDS"

# '~data...name' 必须加引号，否则 bash 波浪号展开
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  --num_processes="$NUM_PROCESSES" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=DPO \
  task_name="$TASK_NAME" \
  model="$MODEL" \
  forget_split="$FORGET_SPLIT" \
  retain_split="$RETAIN_SPLIT" \
  model.model_args.pretrained_model_name_or_path="$MODEL_HF" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_HF" \
  retain_logs_path="$RETAIN_LOGS" \
  paths.output_dir="$CKPT" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
  trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
  trainer.args.learning_rate="$LR" \
  trainer.args.num_train_epochs="$EPOCHS" \
  trainer.args.eval_strategy=no \
  trainer.args.eval_on_start=False \
  trainer.args.gradient_checkpointing=true \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.method_args.beta="$BETA" \
  trainer.method_args.alpha="$ALPHA" \
  data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
  '~data.forget.TOFU_QA_forget.args.hf_args.name' \
  data.forget.TOFU_QA_forget.args.hf_args.path=json \
  "+data.forget.TOFU_QA_forget.args.hf_args.data_files=$DATA_FILE" \
  data.forget.TOFU_QA_forget.args.hf_args.split=train \
  +data.forget.TOFU_QA_forget.args.alternate_key=alternate \
  +data.forget.TOFU_QA_forget.args.return_original=True \
  "${EXTRA_TRAIN_HYDRA[@]}" \
  "${EXTRA_EVAL_HYDRA[@]}" \
  2>&1 | tee -a "$LOG_DIR/${TASK_NAME}.log"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split="$FORGET_SPLIT" holdout_split="$HOLDOUT_SPLIT" \
  model="$MODEL" \
  task_name="$TASK_NAME" \
  model.model_args.pretrained_model_name_or_path="$CKPT" \
  model.tokenizer_args.pretrained_model_name_or_path="$CKPT" \
  paths.output_dir="$CKPT/evals" \
  retain_logs_path="$RETAIN_LOGS" \
  "${EXTRA_EVAL_HYDRA[@]}" \
  2>&1 | tee -a "$LOG_DIR/${TASK_NAME}_eval.log"

INIT_SUMMARY="$SAVES/eval/tofu_${MODEL}_full_local/evals_${FORGET_SPLIT}/TOFU_SUMMARY.json"
if [[ ! -f "$INIT_SUMMARY" ]]; then
  INIT_SUMMARY="saves/eval/tofu_${MODEL}_full/evals_${FORGET_SPLIT}/TOFU_SUMMARY.json"
fi
RETAIN_SUMMARY="${RETAIN_LOGS%/TOFU_EVAL.json}/TOFU_SUMMARY.json"

python scripts/ou_aggregate.py "$CKPT/evals/TOFU_SUMMARY.json" \
  --init-summary "$INIT_SUMMARY" \
  --retain-summary "$RETAIN_SUMMARY" \
  --json "$CKPT/evals/ou_aggregate.json"

python scripts/ou_append_run.py \
  --name "$TASK_NAME" \
  --agg-json "$CKPT/evals/ou_aggregate.json" \
  --method AltPO --hyper "$HYPER" \
  --source selftrain --ckpt-path "$CKPT"

python scripts/record_tofu_result.py \
  --summary "$CKPT/evals/TOFU_SUMMARY.json" \
  --method AltPO \
  --forget-split "$FORGET_SPLIT" \
  --model "$MODEL" \
  --ckpt "$CKPT" \
  --ou-summary "$CKPT/evals/ou_aggregate.json"

echo "[done] AltPO $TASK_NAME hyper=$HYPER -> $CKPT"
df -h "$SAVES" | tail -1
