#!/bin/bash
# E0: audit one representative official checkpoint for each of eight methods.
set -euo pipefail

ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV="${VENV:-/root/autodl-tmp/envs/unlearning}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-$ROOT/configs/specgap/e0_tofu_forget10.yaml}"
P0_RESULT="${P0_RESULT:-$SAVES/specgap/p0_local_simnpo.json}"
RESULT_DIR="${RESULT_DIR:-$SAVES/specgap/e0}"
MODEL_DIR_ROOT="${MODEL_DIR_ROOT:-$SAVES/hf_ckpts}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
FAIL_LOG="${FAIL_LOG:-$ROOT/results/specgap_e0_failures.log}"

ONLY=""
LIMIT=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *) echo "[error] unknown argument: $1"; exit 2 ;;
  esac
done

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

cd "$ROOT"
mkdir -p "$RESULT_DIR" "$MODEL_DIR_ROOT" "$LOG_DIR" "$(dirname "$FAIL_LOG")"

python - "$P0_RESULT" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path) as handle:
        passed = json.load(handle)["gate"]["passed"]
except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
    raise SystemExit(f"[error] invalid or missing P0 result {path}: {error}")
if not passed:
    raise SystemExit("[error] SpecGap P0 did not pass; E0 is blocked")
print(f"[gate] P0 passed: {path}")
PY

mapfile -t SETTINGS < <(python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
checkpoints = cfg["checkpoints"]
methods = [entry["method"] for entry in checkpoints]
repos = [entry["repo_id"] for entry in checkpoints]
if len(checkpoints) != 8 or len(set(methods)) != 8 or len(set(repos)) != 8:
    raise SystemExit("E0 manifest must contain exactly eight unique methods/repos")
print(cfg["draft"])
print(cfg["n"])
print(cfg["seed"])
print(cfg["max_length"])
print(cfg["chunk_size"])
PY
)
DRAFT="${SETTINGS[0]}"
N="${SETTINGS[1]}"
SEED="${SETTINGS[2]}"
MAX_LENGTH="${SETTINGS[3]}"
CHUNK_SIZE="${SETTINGS[4]}"

mapfile -t ENTRIES < <(python - "$CONFIG" "$ONLY" "$LIMIT" <<'PY'
import sys
from omegaconf import OmegaConf

config_path, only, limit = sys.argv[1:4]
cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
entries = [
    item for item in cfg["checkpoints"] if not only or item["method"] == only
]
if limit:
    entries = entries[:int(limit)]
for item in entries:
    print(f"{item['method']}\t{item['repo_id']}")
PY
)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "[info] no E0 checkpoints selected"
  exit 0
fi

i=0
for ENTRY in "${ENTRIES[@]}"; do
  i=$((i + 1))
  IFS=$'\t' read -r METHOD REPO_ID <<<"$ENTRY"
  NAME="${REPO_ID##*/}"
  MODEL_DIR="$MODEL_DIR_ROOT/$NAME"
  OUT="$RESULT_DIR/$NAME.json"

  if [[ "$FORCE" == "0" && -f "$OUT" ]]; then
    echo "[skip] [$i/${#ENTRIES[@]}] $METHOD already audited"
    continue
  fi

  echo "=== [$i/${#ENTRIES[@]}] $METHOD: $NAME ==="
  DOWNLOADED=0
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    DOWNLOADED=1
    mkdir -p "$MODEL_DIR"
    if ! huggingface-cli download "$REPO_ID" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False \
      >>"$LOG_DIR/specgap_download_$NAME.log" 2>&1; then
      echo "$NAME download_failed" >>"$FAIL_LOG"
      rm -rf "$MODEL_DIR"
      echo "[fail] download failed: $REPO_ID"
      continue
    fi
  fi

  if CUDA_VISIBLE_DEVICES="$GPU" python src/specprobe.py \
    --draft "$DRAFT" \
    --target "$MODEL_DIR" \
    --splits forget10 retain90 \
    --n "$N" \
    --seed "$SEED" \
    --max-length "$MAX_LENGTH" \
    --chunk-size "$CHUNK_SIZE" \
    --out "$OUT" \
    2>&1 | tee -a "$LOG_DIR/specgap_$NAME.log"; then
    echo "[ok] $METHOD"
  else
    echo "$NAME audit_failed" >>"$FAIL_LOG"
    rm -f "$OUT"
    echo "[fail] audit failed: $METHOD"
  fi

  if [[ "$DOWNLOADED" == "1" ]]; then
    rm -rf "$MODEL_DIR"
    echo "[cleanup] removed downloaded weights: $MODEL_DIR"
  fi
done

python scripts/specgap_table.py --config "$CONFIG" --result-dir "$RESULT_DIR"
echo "[done] E0 batch finished"
