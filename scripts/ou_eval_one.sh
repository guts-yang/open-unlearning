#!/bin/bash
# 单个官方 ckpt 的「下载 → 评测 → 四维聚合 → 落盘 → 删权重」流程（P1 的基本单元）。
#
# 用法：
#   bash scripts/ou_eval_one.sh <repo_id 或 ckpt 名> [GPU_ID]
#   bash scripts/ou_eval_one.sh open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d0_g0.125_ep10
#
# 关键约定：
#   - 权重下载到 $SAVES/hf_ckpts/<name>，评估完立刻删除（磁盘只有 213G，142 个 ckpt 不能全留）
#   - 评估产物（TOFU_SUMMARY.json + TOFU_EVAL.json）留在 $SAVES/eval/<name>，两者都要留：
#     ou_aggregate.py 需要 TOFU_EVAL.json 里的 forget_truth_ratio.value_by_index 现算 TR
#   - 必须显式传 retain_logs_path，否则 privleak 静默 NaN
#   - 归一化分母与 sMIA 参考默认取 P0-3 产出的 *_local 目录，缺失时回退官方下载日志并告警
set -euo pipefail

REPO="${1:?usage: ou_eval_one.sh <repo_id|ckpt_name> [GPU_ID]}"
GPU="${2:-0}"

# --- 路径（仓库唯一副本在 /usr/local/open-unlearning） ------------------------
ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV=/root/autodl-tmp/envs/unlearning
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
RUNS="$ROOT/results/ou_table3_runs.jsonl"
FAIL_LOG="$ROOT/results/ou_table3_failures.log"

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

NAME="$(basename "$REPO")"
case "$REPO" in
  */*) ;;                       # 已带 org 前缀
  *)   REPO="open-unlearning/$REPO" ;;
esac

MODEL_DIR="$SAVES/hf_ckpts/$NAME"
EVAL_DIR="$SAVES/eval/$NAME"
AGG_JSON="$EVAL_DIR/ou_aggregate.json"

# P0-3 本地产物优先，缺失回退官方下载日志（回退时打印告警）
LOCAL_RETAIN="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_retain90_local"
LOCAL_INIT="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10"
if [[ -f "$LOCAL_RETAIN/TOFU_EVAL.json" ]]; then
  RETAIN_EVAL_DIR="$LOCAL_RETAIN"
else
  RETAIN_EVAL_DIR="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_retain90"
  echo "[warn] 未找到 P0-3 本地 retain90 日志，回退官方下载日志：$RETAIN_EVAL_DIR"
fi
if [[ -f "$LOCAL_INIT/TOFU_SUMMARY.json" ]]; then
  INIT_EVAL_DIR="$LOCAL_INIT"
else
  INIT_EVAL_DIR="$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10"
  echo "[warn] 未找到 P0-3 本地 full 日志，回退官方下载日志：$INIT_EVAL_DIR"
fi
RETAIN_LOGS="$RETAIN_EVAL_DIR/TOFU_EVAL.json"
RETAIN_SUMMARY="$RETAIN_EVAL_DIR/TOFU_SUMMARY.json"
INIT_SUMMARY="$INIT_EVAL_DIR/TOFU_SUMMARY.json"

mkdir -p "$LOG_DIR" "$EVAL_DIR" "$(dirname "$RUNS")"
cd "$ROOT"
echo "=== [$(date +%F\ %T)] $NAME (GPU $GPU) ==="

# --- 1) 下载权重（带重试） ----------------------------------------------------
if [[ -f "$MODEL_DIR/config.json" ]]; then
  echo "[skip] 权重已存在：$MODEL_DIR"
else
  mkdir -p "$MODEL_DIR"
  for attempt in 1 2 3; do
    if huggingface-cli download "$REPO" --local-dir "$MODEL_DIR" \
         --local-dir-use-symlinks False >>"$LOG_DIR/ou_download_$NAME.log" 2>&1; then
      break
    fi
    echo "[warn] 下载失败（第 $attempt 次）：$REPO"
    sleep 10
    [[ $attempt -eq 3 ]] && { echo "$NAME download_failed" >>"$FAIL_LOG"; exit 1; }
  done
fi

# tokenizer：官方 ckpt 一般自带，缺失时回退 target 模型
if [[ -f "$MODEL_DIR/tokenizer.json" ]]; then
  TOK="$MODEL_DIR"
else
  TOK="open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
  echo "[warn] ckpt 无 tokenizer，回退 $TOK"
fi

# --- 2) 评测 ------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="$GPU" python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split=forget10 holdout_split=holdout10 \
  model=Llama-3.2-1B-Instruct \
  task_name="$NAME" \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOK" \
  paths.output_dir="$EVAL_DIR" \
  retain_logs_path="$RETAIN_LOGS" 2>&1 | tee -a "$LOG_DIR/ou_eval_$NAME.log"

if [[ ! -f "$EVAL_DIR/TOFU_SUMMARY.json" ]]; then
  echo "$NAME eval_no_summary" >>"$FAIL_LOG"
  echo "[error] 未产出 TOFU_SUMMARY.json：$EVAL_DIR"
  exit 1
fi

# --- 3) 四维聚合 --------------------------------------------------------------
python scripts/ou_aggregate.py "$EVAL_DIR/TOFU_SUMMARY.json" \
  --init-summary "$INIT_SUMMARY" \
  --retain-summary "$RETAIN_SUMMARY" \
  --json "$AGG_JSON"

# --- 4) 追加一行到 jsonl（供 ou_table.py 生成汇总表） --------------------------
# 官方 ckpt 的 method/hyper 由 ou_append_run.py 从 name 解析；与 TPO run.sh 共用同一份逻辑
python scripts/ou_append_run.py \
  --name "$NAME" --agg-json "$AGG_JSON" --runs "$RUNS" \
  --repo-id "$REPO" --source official

# --- 5) 删权重，只留评估产物 --------------------------------------------------
rm -rf "$MODEL_DIR"
echo "[done] $NAME（权重已删，产物在 $EVAL_DIR）"
df -h "$SAVES" | tail -1
