#!/bin/bash
# SpecGap P0 gate: self-test, then full forget10 vs matched retain90 audit.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,18p' "$0"
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "[error] specgap_p0.sh takes no arguments; configure it with environment variables"
  exit 2
fi

ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV="${VENV:-/root/autodl-tmp/envs/unlearning}"
GPU="${GPU:-0}"
DRAFT="${DRAFT:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
TARGET="${TARGET:-$SAVES/unlearn/tofu_1B_SimNPO_forget10}"
OUT_DIR="${OUT_DIR:-$SAVES/specgap}"
SELF_TEST_OUT="$OUT_DIR/p0_self_test.json"
P0_OUT="$OUT_DIR/p0_local_simnpo.json"

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p "$OUT_DIR"
[[ -f "$TARGET/config.json" ]] || {
  echo "[error] P0 target checkpoint not found: $TARGET"
  exit 2
}

echo "=== SpecGap P0 self-test (draft against itself, n=20) ==="
CUDA_VISIBLE_DEVICES="$GPU" python src/specprobe.py \
  --draft "$DRAFT" \
  --target "$DRAFT" \
  --splits forget10 \
  --n 20 \
  --seed 0 \
  --self-test \
  --out "$SELF_TEST_OUT"

echo "=== SpecGap P0 audit (forget10 full, retain90 matched n=400) ==="
CUDA_VISIBLE_DEVICES="$GPU" python src/specprobe.py \
  --draft "$DRAFT" \
  --target "$TARGET" \
  --splits forget10 retain90 \
  --n 400 \
  --seed 0 \
  --out "$P0_OUT"

python - "$P0_OUT" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path) as handle:
    result = json.load(handle)

forget = result["splits"]["forget10"]["summary"]["mean"]
retain = result["splits"]["retain90"]["summary"]["mean"]
d_value = result["comparison"]["cohens_d"]
passed = forget > retain and d_value > 0.8
result["gate"] = {
    "criterion": "forget_mean > retain_mean and cohens_d > 0.8",
    "passed": passed,
}

temporary = path + ".tmp"
with open(temporary, "w") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
os.replace(temporary, path)

print(
    f"[P0] forget={forget:.6f} retain={retain:.6f} "
    f"Cohen's d={d_value:.4f} passed={passed}"
)
sys.exit(0 if passed else 1)
PY

echo "[pass] SpecGap P0 passed; E0 may proceed. Result: $P0_OUT"
