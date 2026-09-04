#!/bin/bash
# One MUSE unlearn + eval on 2 GPUs (DeepSpeed ZeRO-3, effective batch 32).
#
#   bash scripts/muse_unlearn_one.sh TRAINER [News|Books]
#
# Env: GPU_IDS=0,1 NUM_PROCESSES=2 PER_DEVICE_BS=4 GRAD_ACCUM=4 KEEP_CKPT=0
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
# Weights are local snapshot dirs. Do not set TRANSFORMERS_OFFLINE: datasets
# inherits it and muse-bench/MUSE-Books has no arrow cache until first online load.
unset TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_HUB_OFFLINE

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

trainer="${1:?usage: muse_unlearn_one.sh TRAINER [News|Books]}"
data_split="${2:-News}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA_ARGS=("$@")

GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
KEEP_CKPT="${KEEP_CKPT:-0}"
model="${model:-Llama-2-7b-hf}"

hub_snap() {
  # $1 = models--org--name  OR  org/name
  local spec="$1"
  local dir
  if [[ "$spec" == models--* || "$spec" == datasets--* ]]; then
    dir="$HUGGINGFACE_HUB_CACHE/$spec"
  else
    dir="$HUGGINGFACE_HUB_CACHE/models--${spec//\//--}"
  fi
  if [[ -f "$dir/refs/main" ]]; then
    echo "$dir/snapshots/$(cat "$dir/refs/main")"
    return
  fi
  ls -1d "$dir"/snapshots/* 2>/dev/null | head -1
}

TOK="$(hub_snap NousResearch/Llama-2-7b-hf)"
TARGET="$(hub_snap "muse-bench/MUSE-${data_split}_target")"
if [[ -z "$TARGET" || ! -d "$TARGET" ]]; then
  echo "missing local snapshot for muse-bench/MUSE-${data_split}_target"; exit 1
fi
echo "TARGET=$TARGET"
echo "TOK=$TOK"

case "$data_split" in
  News|Books) ;;
  *) echo "unknown data_split $data_split"; exit 1 ;;
esac

task_name="muse_${model}_${data_split}_${trainer}"
ckpt="$SAVES/unlearn/${task_name}"
log_dir="/root/autodl-tmp/logs"
mkdir -p "$log_dir" "$ckpt"

RETAIN_LOGS="$SAVES/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
if [[ ! -f "$RETAIN_LOGS" ]]; then
  RETAIN_LOGS="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
fi

DS_DIR="$HUGGINGFACE_HUB_CACHE/datasets--muse-bench--MUSE-${data_split}"
if [[ -f "$DS_DIR/refs/main" ]]; then
  DS="$DS_DIR/snapshots/$(cat "$DS_DIR/refs/main")"
else
  DS="$(ls -1d "$DS_DIR"/snapshots/* 2>/dev/null | head -1)"
fi
if [[ -z "$DS" || ! -d "$DS" ]]; then
  echo "missing local dataset snapshot muse-bench/MUSE-${data_split}"; exit 1
fi
echo "DS=$DS"
HYPER="2xA800 ZeRO3 bf16 adamw_torch (bnb paged_adamw unavailable) batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES} lr=1e-5 ep=10"

PDU_ARGS=()
if [[ "$trainer" == "PDU" ]]; then
  if [[ "$data_split" == "News" ]]; then
    eps=1.5
  else
    eps=0.1
  fi
  PDU_ARGS=(
    trainer.method_args.alpha=50
    trainer.method_args.retain_loss_eps="$eps"
    trainer.method_args.dual_step_size=1
    trainer.method_args.dual_warmup_epochs=3
    trainer.method_args.primal_dual=true
  )
  HYPER="$HYPER PDU eps=$eps pref=50 warmup=3"
fi

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$data_split task=$task_name"
echo "GPU_IDS=$GPU_IDS NUM_PROCESSES=$NUM_PROCESSES batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}"

set +e
CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  --num_processes="$NUM_PROCESSES" \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/muse/default \
  model="$model" \
  data_split="$data_split" \
  trainer="$trainer" \
  task_name="$task_name" \
  model.model_args.pretrained_model_name_or_path="$TARGET" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOK" \
  retain_logs_path="$RETAIN_LOGS" \
  paths.output_dir="$ckpt" \
  trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
  trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.do_eval=false \
  trainer.args.eval_on_start=false \
  trainer.args.optim=adamw_torch \
  "${PDU_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  python scripts/record_muse_wmdp_result.py \
    --bench MUSE --split "$data_split" --method "$trainer" --status train_fail \
    --ckpt "$ckpt" --hyper "$HYPER" --note "train exit $train_rc"
  exit "$train_rc"
fi

# Eval metrics load muse-bench/MUSE-* named configs from the hub cache.
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE HF_DATASETS_OFFLINE
set +e
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py experiment=eval/muse/default.yaml \
  data_split="$data_split" \
  task_name="$task_name" \
  model="$model" \
  model.model_args.pretrained_model_name_or_path="$ckpt" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOK" \
  paths.output_dir="$ckpt/evals" \
  retain_logs_path="$RETAIN_LOGS"
eval_rc=$?
set -e

SUMMARY="$ckpt/evals/MUSE_SUMMARY.json"
if [[ "$eval_rc" -ne 0 || ! -f "$SUMMARY" ]]; then
  python scripts/record_muse_wmdp_result.py \
    --bench MUSE --split "$data_split" --method "$trainer" --status eval_fail \
    --summary "$SUMMARY" --ckpt "$ckpt" --hyper "$HYPER" --note "eval exit $eval_rc"
  exit "${eval_rc:-1}"
fi

python scripts/record_muse_wmdp_result.py \
  --bench MUSE --split "$data_split" --method "$trainer" --status ok \
  --summary "$SUMMARY" --ckpt "$ckpt" --hyper "$HYPER"

if [[ "$KEEP_CKPT" != "1" ]]; then
  find "$ckpt" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'optimizer.pt' \) -delete || true
  echo "dropped 7B weights under $ckpt (KEEP_CKPT=0); evals kept"
fi
