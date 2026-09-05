#!/bin/bash
# Sequential AltSpec plan: warmup-only → lr=2e-5 seeds → alt generate → V1 → V2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES
export GPU_IDS="${GPU_IDS:-0,1}"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
echo "[altspec] log=/root/autodl-tmp/logs/altspec_plan.nohup"
echo "[altspec] paths listed in $ROOT/results/altspec_run_paths.md"
echo "[altspec] warmup  $SAVES/unlearn/tofu_1B_SpecDiff_forget10_warmup_only_s0"
echo "[altspec] lr2e5   $SAVES/unlearn/tofu_1B_SpecDiff_forget10_g_lr2e-5_lam1_b0.1_k0.3_ep10_seed1 (and seed2)"
echo "[altspec] alt     $ROOT/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget10/alt5_seed_0.json"
echo "[altspec] v1      $SAVES/unlearn/tofu_1B_SpecDiff_forget10_altspec_v1_wudpo2_seed0"
echo "[altspec] v2      $SAVES/unlearn/tofu_1B_SpecDiff_forget10_altspec_v2_dpo0.5_lr5e-5_seed0 (and lr2e-5)"

bash scripts/specdiff_tofu.sh warmup
bash scripts/specdiff_tofu.sh lr2e5
bash scripts/specdiff_tofu.sh generate
bash scripts/specdiff_tofu.sh v1
bash scripts/specdiff_tofu.sh v2
python scripts/specdiff_table.py
python scripts/specdiff_paper_fig.py
echo "[altspec] plan experiments finished"
