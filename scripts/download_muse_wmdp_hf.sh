#!/bin/bash
# MUSE/WMDP 的 HF 走镜像下载（与 download_muse_wmdp.sh 并行的快通道）。
# 背景：WMDP corpus 走 S3 直连极慢（~20KB/s），不能让它阻塞 HF 项，故拆成两路并行。
# 幂等：huggingface-cli 对已缓存文件只做校验跳过；与主脚本并发安全（hub 有文件锁）。
# 状态：/root/autodl-tmp/logs/download_muse_wmdp_hf.status
set -uo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate

LOG_DIR=/root/autodl-tmp/logs
STATUS="$LOG_DIR/download_muse_wmdp_hf.status"
mkdir -p "$LOG_DIR"

mark () { echo "$(date '+%F %T') [$1] $2" | tee -a "$STATUS"; }
done_ok () { grep -q "\[OK\] $1\$" "$STATUS" 2>/dev/null; }

# --- MUSE 数据集 -----------------------------------------------------------------
for ds in MUSE-News MUSE-Books; do
  done_ok "dataset/$ds" && { mark SKIP "dataset/$ds"; continue; }
  mark RUN "dataset/$ds"
  if huggingface-cli download "muse-bench/$ds" --repo-type dataset \
       >>"$LOG_DIR/dl_$ds.log" 2>&1; then
    mark OK "dataset/$ds"
  else
    mark FAIL "dataset/$ds"
  fi
done

# --- MUSE target 模型（7B，各 ~13G） ----------------------------------------------
for m in MUSE-News_target MUSE-Books_target; do
  done_ok "model/$m" && { mark SKIP "model/$m"; continue; }
  mark RUN "model/$m"
  if huggingface-cli download "muse-bench/$m" >>"$LOG_DIR/dl_$m.log" 2>&1; then
    mark OK "model/$m"
  else
    mark FAIL "model/$m"
  fi
done

# --- zephyr-7b-beta（WMDP 默认模型，~14G） -----------------------------------------
if done_ok model/zephyr; then
  mark SKIP model/zephyr
else
  mark RUN model/zephyr
  if huggingface-cli download HuggingFaceH4/zephyr-7b-beta \
       >>"$LOG_DIR/dl_zephyr.log" 2>&1; then
    mark OK model/zephyr
  else
    mark FAIL model/zephyr
  fi
fi

# --- cais/wmdp MCQ（lm-eval 评测用） ------------------------------------------------
if done_ok wmdp-mcq; then
  mark SKIP wmdp-mcq
else
  mark RUN wmdp-mcq
  if python -c "import datasets; datasets.load_dataset('cais/wmdp', 'wmdp-bio')" \
       >>"$LOG_DIR/dl_wmdp_mcq.log" 2>&1; then
    mark OK wmdp-mcq
  else
    mark FAIL wmdp-mcq
  fi
fi

mark DONE "HF 下载项执行完毕"
df -h /root/autodl-tmp | tail -1
