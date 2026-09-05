#!/bin/bash
# One WMDP-cyber unlearn + lm_eval on 2 GPUs.
# Effective batch stays 16: 1 x 8 accum x 2 GPUs. max_steps=80, lr=5e-5.
#
#   bash scripts/wmdp_unlearn_one.sh TRAINER [cyber]
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
# Zephyr weights are a local snapshot; datasets (mmlu / wmdp) need the hub.
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

trainer="${1:?usage: wmdp_unlearn_one.sh TRAINER [cyber]}"
data_split="${2:-cyber}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")

GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
# default yaml is 1x16 on 1 GPU; keep effective 16 on 2 GPUs
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
KEEP_CKPT="${KEEP_CKPT:-0}"
model="${model:-zephyr-7b-beta}"

hub_snap() {
  local spec="$1"
  local dir="$HUGGINGFACE_HUB_CACHE/models--${spec//\//--}"
  if [[ -f "$dir/refs/main" ]]; then
    echo "$dir/snapshots/$(cat "$dir/refs/main")"
    return
  fi
  ls -1d "$dir"/snapshots/* 2>/dev/null | head -1
}

ZEPHYR="$(hub_snap HuggingFaceH4/zephyr-7b-beta)"
if [[ -z "$ZEPHYR" || ! -d "$ZEPHYR" ]]; then
  echo "missing local snapshot HuggingFaceH4/zephyr-7b-beta"; exit 1
fi
echo "ZEPHYR=$ZEPHYR"

RUN_TAG="${RUN_TAG:-}"
task_name="wmdp_${model}_${data_split}_${trainer}${RUN_TAG:+_${RUN_TAG}}"
ckpt="$SAVES/unlearn/${task_name}"
log_dir="/root/autodl-tmp/logs"
mkdir -p "$log_dir" "$ckpt"

HYPER="2xA800 ZeRO3 bf16 adamw_torch (bnb paged_adamw unavailable) batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES} lr=5e-5 max_steps=80"

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$data_split task=$task_name"
echo "GPU_IDS=$GPU_IDS NUM_PROCESSES=$NUM_PROCESSES batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}"

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  --num_processes="$NUM_PROCESSES" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/wmdp/default \
  model="$model" \
  data_split="$data_split" \
  trainer="$trainer" \
  task_name="$task_name" \
  model.model_args.pretrained_model_name_or_path="$ZEPHYR" \
  model.tokenizer_args.pretrained_model_name_or_path="$ZEPHYR" \
  paths.output_dir="$ckpt" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
  trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.do_eval=false \
  trainer.args.eval_on_start=false \
  trainer.args.optim=adamw_torch \
  "${EXTRA_ARGS[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  python scripts/record_muse_wmdp_result.py \
    --bench WMDP --split "$data_split" --method "$trainer" --status train_fail \
    --ckpt "$ckpt" --hyper "$HYPER" --note "train exit $train_rc"
  exit "$train_rc"
fi
fi

# mmlu is not in the local cache; eval must hit the hub (hf-mirror).
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE HF_DATASETS_OFFLINE
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  EVAL_OVERWRITE="${EVAL_OVERWRITE:-false}"
else
  EVAL_OVERWRITE="${EVAL_OVERWRITE:-true}"
fi

set +e
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py experiment=eval/wmdp/default.yaml \
  data_split="$data_split" \
  task_name="$task_name" \
  model="$model" \
  model.model_args.pretrained_model_name_or_path="$ckpt" \
  model.tokenizer_args.pretrained_model_name_or_path="$ckpt" \
  paths.output_dir="$ckpt/evals" \
  eval.lm_eval.overwrite="$EVAL_OVERWRITE"
eval_rc=$?
set -e

SUMMARY="$ckpt/evals/LMEval_SUMMARY.json"
if [[ ! -f "$SUMMARY" ]]; then
  SUMMARY="$ckpt/evals/LM_EVAL_SUMMARY.json"
fi
if [[ "$eval_rc" -ne 0 || ! -f "$SUMMARY" ]]; then
  python scripts/record_muse_wmdp_result.py \
    --bench WMDP --split "$data_split" --method "$trainer" --status eval_fail \
    --summary "$SUMMARY" --ckpt "$ckpt" --hyper "$HYPER" --note "eval exit $eval_rc"
  exit "${eval_rc:-1}"
fi

python scripts/record_muse_wmdp_result.py \
  --bench WMDP --split "$data_split" --method "$trainer" --status ok \
  --summary "$SUMMARY" --ckpt "$ckpt" --hyper "$HYPER"

if [[ "$KEEP_CKPT" != "1" ]]; then
  find "$ckpt" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'optimizer.pt' \) -delete || true
  echo "dropped 7B weights under $ckpt (KEEP_CKPT=0); evals kept"
fi
