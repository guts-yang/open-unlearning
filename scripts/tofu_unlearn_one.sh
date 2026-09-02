#!/bin/bash
# Train one TOFU unlearning method, then eval, then refresh results/tofu_<model>.md
#
# 用法：
#   bash scripts/tofu_unlearn_one.sh TRAINER [forget_split] [model] [run_tag] [hydra 覆盖...]
#
# 例（P2-T1，与官方同名 ckpt 对齐）：
#   bash scripts/tofu_unlearn_one.sh SimNPO forget10 Llama-3.2-1B-Instruct T1 \
#     trainer.args.learning_rate=1e-5 trainer.method_args.beta=4.5 \
#     trainer.method_args.gamma=0.125 trainer.method_args.delta=0.0
#
# 可覆盖的环境变量（默认值 = 双卡 + 有效 batch 32，与官方设置一致）：
#   GPU_IDS=0,1  NUM_PROCESSES=2  PER_DEVICE_BS=4  GRAD_ACCUM=4  EVAL_GPU=<GPU_IDS 的第一张>
#   SAVES=/root/autodl-tmp/saves
#   EXTRA_HYDRA="model.model_args.attn_implementation=sdpa trainer.args.optim=adamw_torch"
#   # V100：无 FA2；paged_adamw_32bit 依赖的 bitsandbytes 在本环境不可用
# 单卡用法（保持有效 batch 32）：GPU_IDS=0 NUM_PROCESSES=1 GRAD_ACCUM=8 bash scripts/tofu_unlearn_one.sh ...
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

trainer="${1:?usage: tofu_unlearn_one.sh TRAINER [forget_split] [model] [run_tag] [hydra 覆盖...]}"
forget_split="${2:-forget10}"
model="${3:-Llama-3.2-1B-Instruct}"
run_tag="${4:-}"
shift $(( $# < 4 ? $# : 4 ))   # 余下的参数全部当作 hydra 覆盖项
EXTRA_ARGS=("$@")
EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_HYDRA_ARGS <<< "${EXTRA_HYDRA}"
fi

# 训练/并行配置（环境变量可覆盖）
GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"

case "$forget_split" in
  forget10) retain_split=retain90; holdout_split=holdout10 ;;
  forget05) retain_split=retain95; holdout_split=holdout05 ;;
  forget01) retain_split=retain99; holdout_split=holdout01 ;;
  *) echo "unknown forget_split $forget_split"; exit 1 ;;
esac

task_name="tofu_1B_${trainer}_${forget_split}${run_tag:+_${run_tag}}"
ckpt="$SAVES/unlearn/${task_name}"
log_dir="/root/autodl-tmp/logs"
mkdir -p "$log_dir" "$ckpt"
model_hf="open-unlearning/tofu_${model}_full"

# retain 评测 log：优先用 P0-3 产出的本地日志（*_local），否则回退官方下载日志
LOCAL_RETAIN="$SAVES/eval/tofu_${model}_${retain_split}_local/TOFU_EVAL.json"
if [[ -f "$LOCAL_RETAIN" ]]; then
  RETAIN_LOGS="$LOCAL_RETAIN"
else
  RETAIN_LOGS="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
  echo "[warn] 未找到本地 retain 日志 $LOCAL_RETAIN，回退 $RETAIN_LOGS"
fi

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$forget_split task=$task_name"
echo "GPU_IDS=$GPU_IDS NUM_PROCESSES=$NUM_PROCESSES batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}"

CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  --num_processes="$NUM_PROCESSES" \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer="$trainer" \
  task_name="$task_name" \
  model="$model" \
  forget_split="$forget_split" retain_split="$retain_split" \
  model.model_args.pretrained_model_name_or_path="$model_hf" \
  model.tokenizer_args.pretrained_model_name_or_path="$model_hf" \
  retain_logs_path="$RETAIN_LOGS" \
  paths.output_dir="$ckpt" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
  trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split="$forget_split" holdout_split="$holdout_split" \
  model="$model" \
  task_name="$task_name" \
  model.model_args.pretrained_model_name_or_path="$ckpt" \
  model.tokenizer_args.pretrained_model_name_or_path="$ckpt" \
  paths.output_dir="$ckpt/evals" \
  retain_logs_path="$RETAIN_LOGS" \
  "${EXTRA_HYDRA_ARGS[@]}"

# 四维（OU Table 3 口径）：归一化分母用 init-finetuned，sMIA 参考用 retain90
INIT_SUMMARY="$SAVES/eval/tofu_${model}_full_local/evals_forget10/TOFU_SUMMARY.json"
[[ -f "$INIT_SUMMARY" ]] || INIT_SUMMARY="saves/eval/tofu_${model}_full/evals_forget10/TOFU_SUMMARY.json"
RETAIN_SUMMARY="${RETAIN_LOGS%/TOFU_EVAL.json}/TOFU_SUMMARY.json"

python scripts/ou_aggregate.py "$ckpt/evals/TOFU_SUMMARY.json" \
  --init-summary "$INIT_SUMMARY" \
  --retain-summary "$RETAIN_SUMMARY" \
  --json "$ckpt/evals/ou_aggregate.json"

python scripts/record_tofu_result.py \
  --summary "$ckpt/evals/TOFU_SUMMARY.json" \
  --method "$trainer" \
  --forget-split "$forget_split" \
  --model "$model" \
  --ckpt "$ckpt" \
  --ou-summary "$ckpt/evals/ou_aggregate.json"
