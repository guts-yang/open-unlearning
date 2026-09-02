#!/bin/bash
# P0-3：本地评测两个基线模型，产出新口径下的归一化分母与 PrivLeak 参考日志。
#   P0-3a retain90  -> $SAVES/eval/tofu_<model>_retain90_local
#   P0-3b full(初始化微调) -> $SAVES/eval/tofu_<model>_full_local/evals_forget10
#
# 为什么必须本地跑一遍：
#   1. 官方 open-unlearning/eval 的日志里没有 forget_Q_A_PARA_Prob / forget_Q_A_gibberish，
#      Fluency 归一化分母至今为空；
#   2. PrivLeak / sMIA 的参考日志必须与 target 同口径；
#   3. 官方日志是下载来的，不能直接当成本机评测结果写进论文。
#
# 为什么写到 *_local 而不是覆盖官方目录：
#   configs/eval/tofu.yaml 里 overwrite=false，src/evals/base.py 会跳过已存在的指标，
#   写进官方目录会得到「新旧指标混合」的产物，无法用于对照。官方目录保持只读基准。
#
# 用法：
#   bash scripts/ou_eval_baselines.sh            # 两个都跑
#   bash scripts/ou_eval_baselines.sh retain90   # 只跑 retain90
#   bash scripts/ou_eval_baselines.sh full
#   bash scripts/ou_eval_baselines.sh compare    # 只做本地 vs 官方比对
set -euo pipefail

TARGET="${1:-all}"
ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV=/root/autodl-tmp/envs/unlearning
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
GPU="${GPU:-0}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
# V100: EXTRA_HYDRA="model.model_args.attn_implementation=sdpa"
EXTRA_HYDRA_ARGS=()
if [[ -n "${EXTRA_HYDRA:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_HYDRA_ARGS <<< "${EXTRA_HYDRA}"
fi

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"
cd "$ROOT"

MODEL_HF="open-unlearning/tofu_${MODEL}_full"
RETAIN_HF="open-unlearning/tofu_${MODEL}_retain90"
RETAIN_DIR="$SAVES/eval/tofu_${MODEL}_retain90_local"
FULL_DIR="$SAVES/eval/tofu_${MODEL}_full_local/evals_forget10"

eval_one () {   # $1=HF 模型名  $2=输出目录  $3=retain_logs_path（可为空）
  local hf="$1" out="$2" retain_logs="$3"
  mkdir -p "$out"
  echo "=== [$(date +%F\ %T)] eval $hf -> $out ==="
  CUDA_VISIBLE_DEVICES="$GPU" python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split=forget10 holdout_split=holdout10 \
    model="$MODEL" \
    task_name="$(basename "$out")_p0" \
    model.model_args.pretrained_model_name_or_path="$hf" \
    model.tokenizer_args.pretrained_model_name_or_path="$hf" \
    paths.output_dir="$out" \
    retain_logs_path="$retain_logs" \
    "${EXTRA_HYDRA_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/ou_baseline_$(basename "$out").log"
}

case "$TARGET" in
  retain90)
    # retain90 自评：优先用已有的本地日志作参考（重跑时口径一致）；
    # 首次运行时本地日志还不存在，必须回退官方 retain90 日志，
    # 否则 retain_logs_path 指向一个尚不存在的文件，privleak 读不到参考会直接报错。
    if [[ -f "$RETAIN_DIR/TOFU_EVAL.json" ]]; then
      eval_one "$RETAIN_HF" "$RETAIN_DIR" "$RETAIN_DIR/TOFU_EVAL.json"
    else
      echo "[info] 首次评测 retain90，用官方 retain90 日志作 privleak 参考"
      eval_one "$RETAIN_HF" "$RETAIN_DIR" "$SAVES/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json"
    fi
    ;;
  full)
    # full 用本地 retain90 作参考（sMIA 参考系）；没跑过 retain90 就回退官方
    if [[ -f "$RETAIN_DIR/TOFU_EVAL.json" ]]; then
      eval_one "$MODEL_HF" "$FULL_DIR" "$RETAIN_DIR/TOFU_EVAL.json"
    else
      echo "[warn] 本地 retain90 日志不存在，回退官方；建议先跑 retain90"
      eval_one "$MODEL_HF" "$FULL_DIR" "$SAVES/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json"
    fi
    ;;
  compare)
    ;;
  all)
    if [[ -f "$RETAIN_DIR/TOFU_EVAL.json" ]]; then
      eval_one "$RETAIN_HF" "$RETAIN_DIR" "$RETAIN_DIR/TOFU_EVAL.json"
    else
      echo "[info] 首次评测 retain90，用官方 retain90 日志作 privleak 参考"
      eval_one "$RETAIN_HF" "$RETAIN_DIR" "$SAVES/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json"
    fi
    eval_one "$MODEL_HF" "$FULL_DIR" "$RETAIN_DIR/TOFU_EVAL.json"
    ;;
  *) echo "unknown target: $TARGET (all|retain90|full|compare)"; exit 1 ;;
esac

# --- 与官方日志逐项比对（对不上就停在 P0，不进 P1） --------------------------
if [[ "$TARGET" == "all" || "$TARGET" == "compare" ]]; then
  echo "=== 本地 vs 官方对照 ==="
  if python scripts/ou_compare_official.py \
       --local-retain "$RETAIN_DIR" \
       --local-full "$FULL_DIR"; then
    echo "[PASS] 基线对齐官方日志，可以进 P1"
  else
    echo "[FAIL] 基线与官方日志不一致，按 docs/zh/ou-table3-p0.md 的回查路线排查，先别跑 P1"
    exit 1
  fi
fi
