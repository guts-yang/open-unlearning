#!/bin/bash
# 自训矩阵：按官方 ckpt 同名的调参值训练 7 个方法，并全部收进四维汇总表。
#
# 用法：
#   bash scripts/run_selftrain_matrix.sh            # 最小集 7 组（每方法 1 组）
#   bash scripts/run_selftrain_matrix.sh --full     # 完整 10 组（+S2 方差/G2 对照/R2 layer 敏感性）
#   bash scripts/run_selftrain_matrix.sh --only S1  # 只跑某一组
#   bash scripts/run_selftrain_matrix.sh --force    # 忽略已完成记录，重跑
#
# 前置（硬要求）：
#   bash scripts/ou_eval_baselines.sh   # P0-3 必须先跑且通过 ou_compare_official.py，
#                                       # 否则 Fluency 归一化分母为空，四维算不出来。
#
# 为什么需要这个脚本（而不是直接调 tofu_unlearn_one.sh）：
#   tofu_unlearn_one.sh 负责「训练 → 评测 → ou_aggregate --json → 更新 results/tofu_*.md」，
#   但**不会**把结果写进 results/ou_table3_runs.jsonl（四维汇总表的唯一数据源）——
#   因为自训的 task_name 里不含超参串，必须由本脚本显式提供 --method/--hyper 才能与官方
#   ckpt 对拍。所以每组训完由本脚本调 ou_append_run.py 收口。
#
# 可覆盖：GPU_IDS(=0,1) NUM_PROCESSES(=2) SAVES(=/root/autodl-tmp/saves)
#   EXTRA_HYDRA="model.model_args.attn_implementation=sdpa"  # V100 无 FA2，向下传到 tofu_unlearn_one / TPO
set -euo pipefail

ROOT=/usr/local/open-unlearning
cd "$ROOT"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
RUNS="${RUNS:-results/ou_table3_runs.jsonl}"

MODE="min"
ONLY=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) MODE="full"; shift ;;
    --min)  MODE="min";  shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# P0-3 硬前置检查：没有本地基线，Fluency 分母为空，后面全白跑
if [[ ! -f "$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10/TOFU_SUMMARY.json" ]]; then
  echo "[error] 未找到 P0-3 本地 full 基线：$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10/"
  echo "        先跑：bash scripts/ou_eval_baselines.sh"
  exit 1
fi

is_done () {   # $1 = name
  [[ "$FORCE" == "1" ]] && return 1
  [[ -f "$RUNS" ]] || return 1
  grep -q "\"name\": \"$1\"" "$RUNS" 2>/dev/null
}

# 把一组自训结果收进 jsonl（自训必须显式给 method/hyper）
collect () {  # $1=task_name  $2=method  $3=hyper  $4=ckpt_dir
  local agg="$4/evals/ou_aggregate.json"
  if [[ ! -f "$agg" ]]; then
    echo "[warn] 缺少四维聚合结果：$agg（跳过收口）"
    return 0
  fi
  python scripts/ou_append_run.py \
    --name "$1" --agg-json "$agg" \
    --method "$2" --hyper "$3" \
    --source selftrain --ckpt-path "$4"
}

# 跑一组：传 trainer、run_tag、超参串、以及 hydra 覆盖项
run_group () {  # $1=tag $2=trainer $3=hyper $4...=extra hydra 覆盖
  local tag="$1" trainer="$2" hyper="$3"; shift 3
  local task_name="tofu_1B_${trainer}_forget10_${tag}"
  local ckpt="$SAVES/unlearn/${task_name}"

  if [[ -n "$ONLY" && "$tag" != "$ONLY" ]]; then return 0; fi
  if is_done "$task_name"; then
    echo "[skip] $tag 已完成（$task_name 已在 $RUNS）"
    return 0
  fi
  echo "===== [$tag] $trainer  $hyper ====="
  if bash scripts/tofu_unlearn_one.sh "$trainer" forget10 Llama-3.2-1B-Instruct "$tag" "$@"; then
    collect "$task_name" "$trainer" "$hyper" "$ckpt"
  else
    echo "[fail] $tag 训练失败，继续下一组（详见 /root/autodl-tmp/logs/）"
    return 0
  fi
}

# --- 最小集：每方法 1 组 ------------------------------------------------------
run_group S1  SimNPO      lr1e-05_b4.5_a1_d0_g0.125_ep10 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
  trainer.method_args.beta=4.5 trainer.method_args.alpha=1.0 \
  trainer.method_args.delta=0.0 trainer.method_args.gamma=0.125

# RMU 的两个易踩映射：layer -> module_regex（yaml 里没有叫 layer 的参数）
#                    scoeff -> steering_coeff（yaml 默认 2，不是 100）
run_group R1  RMU         lr1e-05_layer10_scoeff100_epoch10 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
  'trainer.method_args.module_regex=model\.layers\.10' \
  trainer.method_args.steering_coeff=100

# GradDiff 的过度遗忘锚点（论文核心论据）
run_group G1  GradDiff    lr1e-05_alpha5_epoch10 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
  trainer.method_args.alpha=5

run_group N1  NPO         lr1e-05_beta0.1_alpha1_epoch10 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
  trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0

# UNDIAL 默认 lr 是 1e-4 不是 1e-5
run_group U1  UNDIAL      lr0.0001_beta10_alpha1_epoch10 \
  trainer.args.learning_rate=1e-4 trainer.args.num_train_epochs=10 \
  trainer.method_args.beta=10 trainer.method_args.alpha=1.0

# GradAscent 下界：官方无 ckpt 可对拍，用仓库默认不调参
run_group GA1 GradAscent  default_lr1e-05_ep10 \
  trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10

# --- TPO：数据 handler 特殊，走社区 run.sh -------------------------------------
run_tpo () {
  local tag="$1" beta="$2" hyper="$3"
  local task_name="tofu_1B_TPO_forget10_${tag}"
  if [[ -n "$ONLY" && "$tag" != "$ONLY" ]]; then return 0; fi
  if is_done "$task_name"; then
    echo "[skip] $tag 已完成"
    return 0
  fi
  echo "===== [$tag] TPO  beta=$beta ====="
  if bash community/methods/TPO/run.sh forget10 "$beta" "$tag"; then
    echo "[ok] $tag（run.sh 内部已调 ou_append_run.py）"
  else
    echo "[fail] $tag 失败，继续下一组"
  fi
}
run_tpo TPO1 0.19 "lr1e-5_beta0.19_alpha0.0_gpt_ep10"

# --- 补充集 -------------------------------------------------------------------
if [[ "$MODE" == "full" ]]; then
  run_group S2  SimNPO      lr2e-05_b3.5_a1_d1_g0.25_ep10 \
    trainer.args.learning_rate=2e-5 trainer.args.num_train_epochs=10 \
    trainer.method_args.beta=3.5 trainer.method_args.alpha=1.0 \
    trainer.method_args.delta=1.0 trainer.method_args.gamma=0.25

  # 弱遗忘对照：与 G1(alpha=5) 构成遗忘强度对比，支撑「过度遗忘锚点」论据
  run_group G2  GradDiff    lr1e-05_alpha1_epoch10 \
    trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
    trainer.method_args.alpha=1

  # RMU layer 敏感性（1B 只有 16 层，layer15 是最后一层风险高，故用 layer5）
  run_group R2  RMU         lr1e-05_layer5_scoeff100_epoch10 \
    trainer.args.learning_rate=1e-5 trainer.args.num_train_epochs=10 \
    'trainer.method_args.module_regex=model\.layers\.5' \
    trainer.method_args.steering_coeff=100
fi

echo "=== 自训矩阵结束 ==="
echo "四维汇总表：python scripts/ou_table.py"
