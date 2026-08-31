#!/bin/bash
# 数据盘分级清理（/root/autodl-tmp）
#
# 默认 dry-run：仅打印将要删除的路径、大小、分级与预计释放总量，不删除任何文件。
# 用法：
#   bash scripts/cleanup_data_disk.sh                 # dry-run，全部候选（L1+L2）
#   bash scripts/cleanup_data_disk.sh --level 1       # dry-run，仅 L1
#   bash scripts/cleanup_data_disk.sh --level 2       # dry-run，L1+L2
#   bash scripts/cleanup_data_disk.sh --apply         # 真正删除（L1+L2）
#   bash scripts/cleanup_data_disk.sh --level 1 --apply
#
# 安全设计：
#   * 删除前校验路径必须以 $BASE/ 开头，否则跳过并告警；
#   * 白名单前缀永不被删（saves/eval、envs、env_hf.sh、hub 内 1B full、logs、search_summary.json 等）；
#   * HF 重复副本（hub 目录之外的 open-unlearning 1B full）不自动删，仅打印提示供人工核对。

set -euo pipefail

BASE="/root/autodl-tmp"
cd "$BASE" 2>/dev/null || { echo "数据盘 $BASE 不存在"; exit 1; }

LEVEL="all"
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --level) : ;;  # 下一参数由 shift 处理（兼容 --level 1 写法）
    --level=*) LEVEL="${arg#*=}" ;;
    --apply) APPLY=1 ;;
    -h|--help) echo "see header comment"; exit 0 ;;
    *) LEVEL="$arg" ;;  # 形如 bash cleanup_data_disk.sh 1
  esac
done

size_of() {
  # 输出人类可读大小；不存在则打印 0
  if [ -e "$1" ]; then
    du -sh --block-size=1 "$1" 2>/dev/null | cut -f1
  else
    echo 0
  fi
}

human() {
  local b="$1"
  awk -v b="$b" 'BEGIN{
    split("B KB MB GB TB",u," ");
    i=1; while(b>=1024 && i<5){b/=1024; i++}
    printf "%.2f %s\n", b, u[i]
  }'
}

# ---------- 白名单（仅展示，永不删除） ----------
WHITELIST=(
  "saves/eval"
  "envs"
  "env_hf.sh"
  "verify_flash_attn.py"
  "dl_flash_attn.sh"
  "huggingface/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full"
  "huggingface/hub/datasets--locuslab--TOFU"
  "logs"
  "saves/mogpu_open_search/search_summary.json"
  "saves/mogpu_gpu_pilot/search_summary.json"
  "candidates"
  "validation"
)

echo "========================================================================="
echo " 数据盘分级清理  BASE=$BASE   LEVEL=$LEVEL   APPLY=$([ $APPLY -eq 1 ] && echo YES || echo NO[dry-run])"
echo "========================================================================="

# ---------- L1 候选（无风险必删） ----------
# 格式: "类型:路径"  类型=d=目录 f=文件
L1_ENTRIES=(
  "d:saves/mogpu_open_search/search"
  "d:saves/mogpu_gpu_pilot/search"
  "d:saves/unlearn/tofu_1B_NPO_forget10"
  "d:npo_record"
  "f:flash_attn-2.8.3*.whl"
)

# ---------- L2 候选（建议删，需确认） ----------
L2_ENTRIES=(
  "d:huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf"
  "d:huggingface/hub/models--locuslab--tofu_ft_llama2-7b"
  "d:paper_models/*/checkpoint-125"
  "f:saves/unlearn/tofu_1B_SimNPO_forget10/model.safetensors"
  "f:saves/unlearn/tofu_1B_RMU_forget10/model.safetensors"
  "f:saves/unlearn/tofu_1B_UNDIAL_forget10/model.safetensors"
  "f:saves/unlearn/tofu_1B_AltPO_forget10/model.safetensors"
)

# HF 重复副本（仅提示，不自动删）
HF_DUP="huggingface/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full"

TOTAL=0
freed() { TOTAL=$((TOTAL + $1)); }

in_whitelist() {
  local p="$1"
  for w in "${WHITELIST[@]}"; do
    case "$p" in
      "$BASE/$w"|"$BASE/$w/"*) return 0 ;;
    esac
  done
  return 1
}

process_entries() {
  local label="$1"; shift
  local entries=("$@")
  echo ""
  echo "----- $label -----"
  local any=0
  for entry in "${entries[@]}"; do
    local typ="${entry%%:*}" pat="${entry#*:}"
    # 展开 glob（bash 数组）
    local expanded=()
    if compgen -G "$BASE/$pat" > /dev/null; then
      expanded=("$BASE"/$pat)
    else
      continue  # 路径不存在，跳过
    fi
    for full in "${expanded[@]}"; do
      any=1
      if in_whitelist "$full"; then
        echo "  [跳过-白名单] $full"
        continue
      fi
      local sz; sz=$(size_of "$full")
      freed "$sz"
      local hs; hs=$(human "$sz")
      if [ $APPLY -eq 1 ]; then
        echo "  [删除] ($hs) $full"
        rm -rf "$full"
      else
        echo "  [将删] ($hs) $full"
      fi
    done
  done
  if [ $any -eq 0 ]; then echo "  (无匹配项)"; fi
  return 0
}

case "$LEVEL" in
  1) process_entries "L1 无风险必删" "${L1_ENTRIES[@]}" ;;
  2|all) process_entries "L1 无风险必删" "${L1_ENTRIES[@]}"
         process_entries "L2 建议删（需确认）" "${L2_ENTRIES[@]}" ;;
  *) echo "未知 --level: $LEVEL (应为 1|2|all)"; exit 1 ;;
esac

echo ""
echo "----- 白名单（受保护，绝不删除） -----"
for w in "${WHITELIST[@]}"; do
  if [ -e "$BASE/$w" ]; then
    printf "  [保护] %s  (%s)\n" "$w" "$(human "$(size_of "$BASE/$w")")"
  fi
done

echo ""
echo "----- HF 重复副本（仅提示，不自动删除） -----"
if [ -e "$BASE/$HF_DUP" ]; then
  printf "  [待核对] %s  (%s)\n" "$HF_DUP" "$(human "$(size_of "$BASE/$HF_DUP")")"
  echo "    请先核对 env_hf.sh 中的 HF_HOME / HF_HUB_CACHE，确认与 hub 内 1B full 为重复副本后再手动删除。"
  if [ -f "$BASE/env_hf.sh" ]; then
    echo "    ---- env_hf.sh 中的缓存路径 ----"
    grep -E "HF_HOME|HF_HUB_CACHE" "$BASE/env_hf.sh" | sed 's/^/      /' || true
  fi
else
  echo "  (不存在，无需处理)"
fi

echo ""
echo "========================================================================="
echo " 预计释放: $(human "$TOTAL")   ($([ $APPLY -eq 1 ] && echo 已执行删除 || echo dry-run 未删除))"
echo "========================================================================="
