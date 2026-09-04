#!/bin/bash
# 一组 WMDP-cyber 遗忘：训 → lm_eval(wmdp_cyber+mmlu) → jsonl。不进 TOFU 四维。
#
# 用法：
#   bash scripts/wmdp_unlearn_one.sh TRAINER [run_tag] [hydra...]
#
# 默认 data_split=cyber。RMU 用 default.yaml 的 layer/down_proj；其它 trainer 覆盖掉 RMU method_args。
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

assert_on_data_disk () {
  local label="$1" path="$2"
  local real
  real="$(readlink -f "$path" 2>/dev/null || true)"
  if [[ -z "$real" || "$real" != /root/autodl-tmp* ]]; then
    echo "[error] $label 必须在 /root/autodl-tmp 下: path=$path real=${real:-missing}"
    exit 1
  fi
}

assert_on_data_disk HF_HOME "${HF_HOME:?source env_hf.sh}"
assert_on_data_disk saves "$ROOT/saves"
assert_on_data_disk "data/wmdp" "$ROOT/data/wmdp"

for need in cyber-forget-corpus.jsonl cyber-retain-corpus.jsonl; do
  if [[ ! -s "$ROOT/data/wmdp/wmdp-corpora/$need" ]]; then
    echo "[error] 缺少 $ROOT/data/wmdp/wmdp-corpora/$need"
    exit 1
  fi
done

trainer="${1:?usage: wmdp_unlearn_one.sh TRAINER [run_tag] [hydra...]}"
run_tag="${2:-}"
if [[ $# -ge 2 ]]; then shift 2; else shift $#; fi
EXTRA_ARGS=("$@")

data_split="${WMDP_SPLIT:-cyber}"
GPU_IDS="${GPU_IDS:-0,1}"
_cgroup_gib=999
if [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  _cgroup_gib=$(($(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) / 1024 / 1024 / 1024))
fi
if [[ -z "${NUM_PROCESSES:-}" ]]; then
  if [[ "$_cgroup_gib" -le 64 ]]; then
    NUM_PROCESSES=1
    GPU_IDS="${GPU_IDS%%,*}"
    echo "[info] cgroup RAM ${_cgroup_gib}GiB≤64，单进程以免双份 7B 被 OOM killer"
  else
    NUM_PROCESSES=2
  fi
fi
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
# 有效 batch 16（与 default.yaml 一致）：2 卡 1×16×1 不对，原 1×16×2=32；计划写有效 16。
# 保持 1×16×N 的 per-step 语义：1 卡时仍 accum=16（有效 16）。
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
MODEL_WEIGHTS="HuggingFaceH4/zephyr-7b-beta"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"

if [[ -z "${EXTRA_HYDRA+x}" ]]; then
  EXTRA_HYDRA="model.model_args.attn_implementation=sdpa trainer.args.optim=adamw_torch model.model_args.low_cpu_mem_usage=true"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q V100; then
    if [[ "$ACCELERATE_CONFIG" == "configs/accelerate/default_config.yaml" ]]; then
      ACCELERATE_CONFIG="configs/accelerate/v100_single.yaml"
    fi
    EXTRA_HYDRA="$EXTRA_HYDRA model.model_args.torch_dtype=float16 trainer.args.bf16=false trainer.args.bf16_full_eval=false trainer.args.fp16=false trainer.args.optim=adafactor"
    echo "[info] V100: fp16 Adafactor + $ACCELERATE_CONFIG（无 DeepSpeed Adam）"
  fi
fi
EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_HYDRA_ARGS <<< "${EXTRA_HYDRA}"
fi
EXTRA_TRAIN_HYDRA=()
EXTRA_EVAL_HYDRA=()
for _h in "${EXTRA_HYDRA_ARGS[@]}"; do
  if [[ "$_h" == trainer.* || "$_h" == +trainer.* ]]; then
    EXTRA_TRAIN_HYDRA+=("$_h")
  else
    EXTRA_EVAL_HYDRA+=("$_h")
  fi
done

task_name="wmdp_zephyr_${data_split}_${trainer}${run_tag:+_${run_tag}}"
ckpt="$SAVES/unlearn/${task_name}"
mkdir -p "$LOG_DIR" "$ckpt"

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$data_split task=$task_name"

# trainer=X 会整份替换 default 里的 RMU method_args，无需再清 module_regex。
HYPER="lr5e-5_maxsteps80_bs16_constant"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q V100; then
  HYPER="${HYPER}_v100_fp16_adafactor_np${NUM_PROCESSES}"
fi

train_ok=1
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
    --config_file "$ACCELERATE_CONFIG" --main_process_port "$MASTER_PORT" \
    --num_processes="$NUM_PROCESSES" \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/wmdp/default.yaml \
    trainer="$trainer" \
    task_name="$task_name" \
    model=zephyr-7b-beta \
    data_split="$data_split" \
    model.model_args.pretrained_model_name_or_path="$MODEL_WEIGHTS" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL_WEIGHTS" \
    paths.output_dir="$ckpt" \
    trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
    trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.args.learning_rate=5e-5 \
    trainer.args.max_steps=80 \
    trainer.args.lr_scheduler_type=constant \
    trainer.args.save_strategy=no \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=false \
    trainer.args.do_eval=false \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    "${EXTRA_TRAIN_HYDRA[@]}" \
    "${EXTRA_EVAL_HYDRA[@]}" \
    "${EXTRA_ARGS[@]}"; then
    train_ok=0
  fi
fi

eval_ok=1
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  eval_ok=1
  NOTE="${NOTE:+$NOTE; }skip_eval"
elif [[ "$train_ok" == "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py \
    experiment=eval/wmdp/default.yaml \
    data_split="$data_split" \
    task_name="$task_name" \
    model=zephyr-7b-beta \
    model.model_args.pretrained_model_name_or_path="$ckpt" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL_WEIGHTS" \
    paths.output_dir="$ckpt/evals" \
    "${EXTRA_EVAL_HYDRA[@]}"; then
    eval_ok=0
  fi
else
  eval_ok=0
fi

STATUS=ok
NOTE="${NOTE:-}"
if [[ "$train_ok" != "1" ]]; then
  STATUS=train_fail
  NOTE="train failed"
elif [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  NOTE="${NOTE:-skip_eval}"
elif [[ "$eval_ok" != "1" ]]; then
  STATUS=eval_fail
  NOTE="eval failed"
fi

SUMMARY="$ckpt/evals/LMEval_SUMMARY.json"
python scripts/record_muse_wmdp_result.py \
  --summary "$SUMMARY" \
  --benchmark wmdp \
  --method "$trainer" \
  --data-split "$data_split" \
  --task-name "$task_name" \
  --ckpt "$ckpt" \
  --hyper "$HYPER" \
  --status "$STATUS" \
  --note "$NOTE"

if [[ "${KEEP_CKPT:-0}" != "1" && "$eval_ok" == "1" ]]; then
  echo "[info] KEEP_CKPT=0，删除权重，保留 evals"
  find "$ckpt" -maxdepth 1 -type f ! -name '*.json' -delete || true
  rm -rf "$ckpt/checkpoint-"* "$ckpt/global_step"* "$ckpt/logs" 2>/dev/null || true
fi

[[ "$STATUS" == "ok" ]]
