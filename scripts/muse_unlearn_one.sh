#!/bin/bash
# 一组 MUSE 遗忘：训 → 评 → 写入 results/muse_wmdp_runs.jsonl（不进 TOFU 四维）。
#
# 用法：
#   bash scripts/muse_unlearn_one.sh TRAINER [News|Books] [run_tag] [hydra...]
#
# 钉死：target=muse-bench/MUSE-${split}_target，tokenizer=NousResearch/Llama-2-7b-hf
# 超参：lr=1e-5 constant 10 epoch，有效 batch 32（4×4×2）。
# V100：默认 sdpa + adamw_torch（paged_adamw_32bit / FA2 不可用）。
# KEEP_CKPT=1 评完保留权重；默认评完删 ckpt 只留 evals/。
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
assert_on_data_disk logs "$ROOT/logs"

trainer="${1:?usage: muse_unlearn_one.sh TRAINER [News|Books] [run_tag] [hydra...]}"
data_split="${2:-News}"
run_tag="${3:-}"
if [[ $# -ge 3 ]]; then shift 3; else shift $#; fi
EXTRA_ARGS=("$@")

case "$data_split" in
  News|Books) ;;
  *) echo "data_split must be News or Books"; exit 1 ;;
esac

GPU_IDS="${GPU_IDS:-0,1}"
# 未显式指定进程数时：容器 cgroup RAM≤64G 则单进程（双 rank 各载一份 7B 会 SIGKILL）。
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
# V100 单卡 seq=2048 时 bs=4 会在 backward OOM；1×32×1 仍有效 32。
if [[ -z "${PER_DEVICE_BS:-}" ]]; then
  if [[ "$NUM_PROCESSES" -eq 1 ]]; then PER_DEVICE_BS=1; else PER_DEVICE_BS=4; fi
fi
if [[ -z "${GRAD_ACCUM:-}" ]]; then
  if [[ "$NUM_PROCESSES" -eq 1 ]]; then GRAD_ACCUM=32; else GRAD_ACCUM=4; fi
fi
EVAL_GPU="${EVAL_GPU:-${GPU_IDS%%,*}}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
TOKENIZER="${TOKENIZER:-NousResearch/Llama-2-7b-hf}"
MODEL_WEIGHTS="muse-bench/MUSE-${data_split}_target"
RETAIN_LOGS="$SAVES/eval/muse_Llama-2-7b-hf_${data_split}_retrain/MUSE_EVAL.json"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"

# V100 无 bf16/FA2/paged_adamw；用 fp16 + ZeRO3 optimizer CPU offload。
if [[ -z "${EXTRA_HYDRA+x}" ]]; then
  EXTRA_HYDRA="model.model_args.attn_implementation=sdpa trainer.args.optim=adamw_torch model.model_args.low_cpu_mem_usage=true"
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q V100; then
    if [[ "$ACCELERATE_CONFIG" == "configs/accelerate/default_config.yaml" ]]; then
      # 不用 ZeRO3：CPU Adam 状态 ~56GiB，超过 AutoDL cgroup 50GiB。
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

task_name="muse_Llama-2-7b-hf_${data_split}_${trainer}${run_tag:+_${run_tag}}"
ckpt="$SAVES/unlearn/${task_name}"
mkdir -p "$LOG_DIR" "$ckpt"

if [[ ! -f "$RETAIN_LOGS" ]]; then
  echo "[error] 缺少 retrain 对照 $RETAIN_LOGS"
  exit 1
fi

export MASTER_PORT
MASTER_PORT="$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")"
echo "MASTER_PORT=$MASTER_PORT trainer=$trainer split=$data_split task=$task_name"
echo "weights=$MODEL_WEIGHTS tokenizer=$TOKENIZER"
echo "GPU_IDS=$GPU_IDS batch=${PER_DEVICE_BS}x${GRAD_ACCUM}x${NUM_PROCESSES}"

HYPER="lr1e-5_ep10_bs32_constant"
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q V100; then
  HYPER="${HYPER}_v100_fp16_adafactor_np${NUM_PROCESSES}"
fi
if [[ "$trainer" == "PDU" ]]; then
  if [[ "$data_split" == "News" ]]; then
    PDU_EPS="${PDU_EPS:-1.5}"
  else
    PDU_EPS="${PDU_EPS:-0.1}"
  fi
  PDU_ARGS=(
    trainer.method_args.gamma=1.0
    trainer.method_args.alpha=50
    trainer.method_args.primal_dual=true
    trainer.method_args.retain_loss_eps="$PDU_EPS"
    trainer.method_args.dual_step_size=1
    trainer.method_args.dual_update_upon=step
    trainer.method_args.dual_warmup_epochs=3
  )
  HYPER="${HYPER}_pdu_eps${PDU_EPS}_pref50"
else
  PDU_ARGS=()
fi

train_ok=1
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
    --config_file "$ACCELERATE_CONFIG" --main_process_port "$MASTER_PORT" \
    --num_processes="$NUM_PROCESSES" \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default.yaml \
    trainer="$trainer" \
    task_name="$task_name" \
    model=Llama-2-7b-hf \
    data_split="$data_split" \
    model.model_args.pretrained_model_name_or_path="$MODEL_WEIGHTS" \
    model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER" \
    retain_logs_path="$RETAIN_LOGS" \
    paths.output_dir="$ckpt" \
    trainer.args.per_device_train_batch_size="$PER_DEVICE_BS" \
    trainer.args.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.args.learning_rate=1e-5 \
    trainer.args.num_train_epochs=10 \
    trainer.args.lr_scheduler_type=constant \
    trainer.args.save_strategy=no \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=false \
    trainer.args.do_eval=false \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    "${PDU_ARGS[@]}" \
    "${EXTRA_TRAIN_HYDRA[@]}" \
    "${EXTRA_EVAL_HYDRA[@]}" \
    "${EXTRA_ARGS[@]}"; then
    train_ok=0
    echo "[error] 训练失败 $task_name"
  fi
fi

eval_ok=1
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  eval_ok=1
  NOTE="${NOTE:+$NOTE; }skip_eval"
elif [[ "$train_ok" == "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$EVAL_GPU" python src/eval.py \
    experiment=eval/muse/default.yaml \
    data_split="$data_split" \
    task_name="$task_name" \
    model=Llama-2-7b-hf \
    model.model_args.pretrained_model_name_or_path="$ckpt" \
    model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER" \
    paths.output_dir="$ckpt/evals" \
    retain_logs_path="$RETAIN_LOGS" \
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

SUMMARY="$ckpt/evals/MUSE_SUMMARY.json"
if [[ ! -f "$SUMMARY" ]]; then
  SUMMARY="$ckpt/evals/MUSE_EVAL.json"
fi
python scripts/record_muse_wmdp_result.py \
  --summary "$SUMMARY" \
  --benchmark muse \
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
