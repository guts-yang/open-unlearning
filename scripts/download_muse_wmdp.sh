#!/bin/bash
# MUSE / WMDP 下载准备（全部落数据盘，符合 .codebuddy/CONSTRAINTS.md 顶级约束）。
#
# 用法（后台执行，日志与状态落数据盘）：
#   nohup bash scripts/download_muse_wmdp.sh > /root/autodl-tmp/logs/download_muse_wmdp.log 2>&1 &
#   tail -f /root/autodl-tmp/logs/download_muse_wmdp.status     # 看逐项进度
#
# 幂等：已标记 OK 的项自动跳过，可直接重跑补漏。
# 明细：docs/zh/【2】muse-wmdp-prep.md 第五节。
set -uo pipefail   # 不用 -e：单项失败标记后继续下一项

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate

ROOT=/usr/local/open-unlearning
LOG_DIR=/root/autodl-tmp/logs
STATUS="$LOG_DIR/download_muse_wmdp.status"
mkdir -p "$LOG_DIR"
cd "$ROOT"

mark () { echo "$(date '+%F %T') [$1] $2" | tee -a "$STATUS"; }

done_ok () { grep -q "^\S* \[OK\] $1\$" "$STATUS" 2>/dev/null; }

# --- 1) WMDP corpus（wget S3 + 密码 unzip；data/wmdp 已软链到数据盘） -----------
if done_ok wmdp-corpus; then
  mark SKIP wmdp-corpus
else
  mark RUN wmdp-corpus
  if python setup_data.py --wmdp >>"$LOG_DIR/dl_wmdp_corpora.log" 2>&1; then
    mark OK wmdp-corpus
  else
    mark FAIL wmdp-corpus
  fi
fi

# --- 2-3) MUSE 数据集 -----------------------------------------------------------
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

# --- 4-5) MUSE target 模型（7B，各 ~13G） ---------------------------------------
for m in MUSE-News_target MUSE-Books_target; do
  done_ok "model/$m" && { mark SKIP "model/$m"; continue; }
  mark RUN "model/$m"
  if huggingface-cli download "muse-bench/$m" >>"$LOG_DIR/dl_$m.log" 2>&1; then
    mark OK "model/$m"
  else
    mark FAIL "model/$m"
  fi
done

# --- 6) zephyr-7b-beta（WMDP 默认模型，~14G） ------------------------------------
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

# --- 7) cais/wmdp MCQ（lm-eval 评测用） ------------------------------------------
if done_ok wmdp-mcq; then
  mark SKIP wmdp-mcq
else
  mark RUN wmdp-mcq
  if python -c "import datasets; datasets.load_dataset('cais/wmdp', 'wmdp_bio')" \
       >>"$LOG_DIR/dl_wmdp_mcq.log" 2>&1; then
    mark OK wmdp-mcq
  else
    mark FAIL wmdp-mcq
  fi
fi

mark DONE "全部下载项执行完毕（详见 $STATUS）"
echo "---- 磁盘 ----"
df -h /root/autodl-tmp | tail -1
