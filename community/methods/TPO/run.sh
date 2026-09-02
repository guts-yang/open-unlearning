#!/bin/bash
# TPO (Targeted Preference Optimization) on TOFU，已接入本仓库的四维评测链路。
# Paper: "Not All Tokens Are Meant to Be Forgotten" (AAAI 2026, arXiv:2506.03142)
# Official code: https://github.com/guts-yang/Unlearning-TPO
#
# 用法：
#   bash community/methods/TPO/run.sh                        # 默认 forget10 + beta 0.19
#   bash community/methods/TPO/run.sh forget10 0.19 TPO1     # 显式 split / beta / run_tag
#   BETA=0.30 bash community/methods/TPO/run.sh              # 环境变量覆盖 beta
#   bash community/methods/TPO/run.sh forget10 0.23 TPO_b23  # 补搜 beta
#
# 可覆盖的环境变量：
#   GPU_IDS=0,1  NUM_PROCESSES=2  SAVES=/root/autodl-tmp/saves  CLASSIFIER=gpt
#   LR=1e-5  EPOCHS=10  PER_DEVICE_BS=4  GRAD_ACCUM=4
#   # 有效 batch = 4x4x2 = 32（与官方 / 其他方法一致；V100 单卡 16 会 OOM）
#   EXTRA_HYDRA="model.model_args.attn_implementation=sdpa trainer.args.optim=adamw_torch"
#
# 前置：
#   1. P0-3 本地基线已跑（retain90_local / full_local），否则 Fluency 分母缺失、
#      retain 日志会回退到官方下载日志（会打告警，且 sMIA 参考口径不一致）。
#   2. 标注数据在 community/methods/TPO/data/（已入库；丢了就跑 prepare_data.py 重建）。
#
# 与其他方法的一致性：训练 → 评测 → ou_aggregate.py --json → ou_append_run.py，
# 四维结果进 results/ou_table3_runs.jsonl，可直接被 ou_table.py 汇总。
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

FORGET_SPLIT="${1:-forget10}"
BETA="${BETA:-${2:-0.19}}"
RUN_TAG="${3:-}"

case "$FORGET_SPLIT" in
  forget10) RETAIN_SPLIT=retain90; HOLDOUT_SPLIT=holdout10 ;;
  forget05) RETAIN_SPLIT=retain95; HOLDOUT_SPLIT=holdout05 ;;
  forget01) RETAIN_SPLIT=retain99; HOLDOUT_SPLIT=holdout01 ;;
  *) echo "unknown forget_split: $FORGET_SPLIT"; exit 1 ;;
esac

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
CLASSIFIER="${CLASSIFIER:-gpt}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_HYDRA_ARGS <<< "${EXTRA_HYDRA}"
fi
EXTRA_TRAIN_HYDRA=()
EXTRA_EVAL_HYDRA=()
for _h in "${EXTRA_HYDRA_ARGS[@]}"; do
  if [[ "$_h" == trainer.* ]]; then
    EXTRA_TRAIN_HYDRA+=("$_h")
  else
    EXTRA_EVAL_HYDRA+=("$_h")
  fi
done

TASK_NAME="tofu_1B_TPO_${FORGET_SPLIT}${RUN_TAG:+_${RUN_TAG}}"
CKPT="$SAVES/unlearn/$TASK_NAME"
MODEL_HF="open-unlearning/tofu_${MODEL}_full"
DATA_FILE="community/methods/TPO/data/${FORGET_SPLIT}_with_common_words_${CLASSIFIER}.json"
# 超参串用于汇总表与对账（HF 上没有 TPO 的官方 ckpt，无同名可对拍）
HYPER="lr${LR}_beta${BETA}_alpha0_${CLASSIFIER}_ep${EPOCHS}"

mkdir -p "$LOG_DIR" "$CKPT"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "[error] TPO 标注数据缺失：$DATA_FILE"
  echo "        重建：python community/methods/TPO/prepare_data.py --src <Unlearning-TPO 仓库路径>"
  exit 1
fi

# retain 评测日志：优先 P0-3 本地产物，缺失回退官方下载日志并告警
LOCAL_RETAIN="$SAVES/eval/tofu_${MODEL}_${RETAIN_SPLIT}_local/TOFU_EVAL.json"
if [[ -f "$LOCAL_RETAIN" ]]; then
  RETAIN_LOGS="$LOCAL_RETAIN"
else
  RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
  echo "[warn] 未找到 P0-3 本地 retain 日志，回退 $RETAIN_LOGS（sMIA 参考口径可能不一致）"
fi

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"

echo "=== [$(date +%F\ %T)] TPO: split=$FORGET_SPLIT beta=$BETA classifier=$CLASSIFIER ==="
echo "    task=$TASK_NAME  ckpt=$CKPT  batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}  GPUs=$GPU_IDS"

# --- 训练：Accelerate + DeepSpeed ZeRO-3（与 tofu_unlearn_one.sh 同配置） ------
# 注意：'~data...name' 必须加引号，否则 bash 会对 ~ 做波浪号展开
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  --num_processes="$NUM_PROCESSES" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=TPO \
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
  trainer.method_args.alpha=0.0 \
  data.forget.TOFU_QA_forget.handler=QAwithCommonWordsDataset \
  '~data.forget.TOFU_QA_forget.args.hf_args.name' \
  data.forget.TOFU_QA_forget.args.hf_args.path=json \
  "+data.forget.TOFU_QA_forget.args.hf_args.data_files=$DATA_FILE" \
  data.forget.TOFU_QA_forget.args.hf_args.split=train \
  "${EXTRA_TRAIN_HYDRA[@]}" \
  "${EXTRA_EVAL_HYDRA[@]}" \
  2>&1 | tee -a "$LOG_DIR/${TASK_NAME}.log"

# --- 评测 --------------------------------------------------------------------
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

# --- 四维聚合 + 落盘 ----------------------------------------------------------
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
  --method TPO --hyper "$HYPER" \
  --source selftrain --ckpt-path "$CKPT"

python scripts/record_tofu_result.py \
  --summary "$CKPT/evals/TOFU_SUMMARY.json" \
  --method TPO \
  --forget-split "$FORGET_SPLIT" \
  --model "$MODEL" \
  --ckpt "$CKPT" \
  --ou-summary "$CKPT/evals/ou_aggregate.json"

echo "[done] TPO $TASK_NAME -> $CKPT"
df -h "$SAVES" | tail -1
