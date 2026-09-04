#!/bin/bash
# E0: audit local-best checkpoints first, then original official anchors.
#
# Pipeline: background prefetch downloads; GPUs only take ready weights
# (config.json + model weights on disk). Already-downloaded checkpoints start
# immediately — no waiting for other downloads to finish.
#
# Default: GPUS=0,1. Set GPUS=0 for single-GPU.
set -euo pipefail

ROOT=/usr/local/open-unlearning
SAVES="${SAVES:-/root/autodl-tmp/saves}"
VENV="${VENV:-/root/autodl-tmp/envs/unlearning}"
GPUS="${GPUS:-0,1}"
PREFETCH="${PREFETCH:-2}"   # concurrent background downloads
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
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --prefetch) PREFETCH="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,24p' "$0"
      exit 0
      ;;
    *) echo "[error] unknown argument: $1"; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "[error] GPUS is empty"; exit 2
fi

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
if len(checkpoints) != 8 or len(set(methods)) != 8:
    raise SystemExit("E0 manifest must contain exactly eight unique methods")
if len(set(repos)) != len(repos):
    raise SystemExit("E0 preferred repo_id values must be unique")
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
import os
import sys
from omegaconf import OmegaConf

config_path, only, limit = sys.argv[1:4]
cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def has_weights(path):
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(path, name)):
            return True
    # sharded checkpoints
    for entry in os.listdir(path) if os.path.isdir(path) else []:
        if entry.endswith(".safetensors") or entry.startswith("pytorch_model-"):
            return True
    return False


entries = []
seen = set()
for item in cfg["checkpoints"]:
    if only and item["method"] != only:
        continue
    candidates = []
    local_path = item.get("local_path")
    if local_path:
        candidates.append(
            {
                "kind": "local",
                "method": item["method"],
                "source": item.get("local_name") or os.path.basename(local_path.rstrip("/")),
                "repo_id": item["repo_id"],
                "target": local_path,
                "download": False,
            }
        )
    candidates.append(
        {
            "kind": "preferred",
            "method": item["method"],
            "source": item["repo_id"].rsplit("/", 1)[-1],
            "repo_id": item["repo_id"],
            "target": item["repo_id"],
            "download": True,
        }
    )
    official = item.get("official_repo_id")
    if official and official != item["repo_id"]:
        candidates.append(
            {
                "kind": "official",
                "method": item["method"],
                "source": official.rsplit("/", 1)[-1],
                "repo_id": official,
                "target": official,
                "download": True,
            }
        )
    picked_preferred = False
    for candidate in candidates:
        key = candidate["source"]
        if key in seen:
            continue
        if candidate["kind"] == "local" and not has_weights(candidate["target"]):
            print(
                f"[skip-missing-local] {candidate['method']} {candidate['target']}",
                file=sys.stderr,
            )
            continue
        if candidate["kind"] == "preferred" and picked_preferred:
            continue
        if candidate["kind"] == "local":
            picked_preferred = True
        elif candidate["kind"] == "preferred":
            picked_preferred = True
        seen.add(key)
        entries.append(candidate)

if limit:
    entries = entries[: int(limit)]
for item in entries:
    print(
        f"{item['method']}\t{item['kind']}\t{item['source']}\t"
        f"{item['repo_id']}\t{item['target']}\t{int(item['download'])}"
    )
PY
)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
  echo "[info] no E0 checkpoints selected"
  exit 0
fi

# Resolve each entry to a concrete model_dir and whether it needs HF download.
declare -a JOBS=()   # tab fields: idx method kind name repo_id model_dir needs_dl
total=${#ENTRIES[@]}
i=0
for ENTRY in "${ENTRIES[@]}"; do
  i=$((i + 1))
  IFS=$'\t' read -r METHOD KIND NAME REPO_ID TARGET DOWNLOAD <<<"$ENTRY"
  if [[ "$KIND" == "local" ]]; then
    MODEL_DIR="$TARGET"
    NEEDS_DL=0
  else
    MODEL_DIR="$MODEL_DIR_ROOT/$NAME"
    NEEDS_DL=1
  fi
  JOBS+=("$i"$'\t'"$METHOD"$'\t'"$KIND"$'\t'"$NAME"$'\t'"$REPO_ID"$'\t'"$MODEL_DIR"$'\t'"$NEEDS_DL")
done

echo "[schedule] ${#JOBS[@]} jobs | GPUs=${GPUS} | prefetch=${PREFETCH}"

# Ready = config + at least one weight file (config alone appears mid-download).
is_ready() {
  local dir="$1"
  [[ -f "$dir/config.json" ]] || return 1
  [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" ]] && return 0
  compgen -G "$dir/*.safetensors" >/dev/null && return 0
  compgen -G "$dir/pytorch_model-*.bin" >/dev/null && return 0
  return 1
}

is_done() {
  local name="$1"
  [[ "$FORCE" == "0" && -f "$RESULT_DIR/$name.json" ]]
}

download_one() {
  local name="$1" repo_id="$2" model_dir="$3"
  if is_ready "$model_dir"; then
    echo "[prefetch] already ready: $name"
    return 0
  fi
  echo "[prefetch] downloading $name"
  mkdir -p "$model_dir"
  if huggingface-cli download "$repo_id" \
    --local-dir "$model_dir" \
    --local-dir-use-symlinks False \
    >>"$LOG_DIR/specgap_download_$name.log" 2>&1; then
    echo "[prefetch] ready $name"
    return 0
  fi
  echo "$name download_failed" >>"$FAIL_LOG"
  echo "[fail] download failed: $repo_id"
  rm -rf "$model_dir"
  return 1
}

audit_ready() {
  local idx="$1" method="$2" kind="$3" name="$4"
  local model_dir="$5" gpu="$6" needs_dl="$7"
  local out="$RESULT_DIR/$name.json"

  if is_done "$name"; then
    echo "[skip] [$idx/$total] $method ($kind) already audited"
    return 0
  fi
  if ! is_ready "$model_dir"; then
    echo "[fail] [$idx/$total] $name not ready on disk: $model_dir"
    echo "$name not_ready" >>"$FAIL_LOG"
    return 1
  fi

  echo "=== [$idx/$total] GPU=$gpu $method ($kind): $name ==="
  if CUDA_VISIBLE_DEVICES="$gpu" python src/specprobe.py \
    --draft "$DRAFT" \
    --target "$model_dir" \
    --device cuda:0 \
    --splits forget10 retain90 \
    --n "$N" \
    --seed "$SEED" \
    --max-length "$MAX_LENGTH" \
    --chunk-size "$CHUNK_SIZE" \
    --out "$out" \
    2>&1 | tee -a "$LOG_DIR/specgap_$name.log"; then
    echo "[ok] GPU=$gpu $method $kind"
    # Only delete HF downloads we staged; never delete local selftrain dirs.
    if [[ "$needs_dl" == "1" ]]; then
      rm -rf "$model_dir"
      echo "[cleanup] removed downloaded weights: $model_dir"
    fi
    return 0
  fi
  echo "$name audit_failed" >>"$FAIL_LOG"
  rm -f "$out"
  echo "[fail] audit failed: $method $kind"
  return 1
}

# ---- background prefetch ----
declare -A DL_PID=()
declare -A DL_NAME=()

start_prefetch() {
  local name="$1" repo_id="$2" model_dir="$3"
  # already downloading or ready?
  if is_ready "$model_dir"; then
    return 0
  fi
  for pid in "${!DL_PID[@]}"; do
    if [[ "${DL_NAME[$pid]}" == "$name" ]]; then
      return 0
    fi
  done
  # wait if at prefetch capacity
  while true; do
    local live=0 pid
    for pid in "${!DL_PID[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        live=$((live + 1))
      else
        wait "$pid" 2>/dev/null || true
        unset "DL_PID[$pid]"
        unset "DL_NAME[$pid]"
      fi
    done
    [[ $live -lt $PREFETCH ]] && break
    sleep 2
  done
  download_one "$name" "$repo_id" "$model_dir" &
  local pid=$!
  DL_PID[$pid]=1
  DL_NAME[$pid]="$name"
}

reap_downloads() {
  local pid
  for pid in "${!DL_PID[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      unset "DL_PID[$pid]"
      unset "DL_NAME[$pid]"
    fi
  done
}

# Kick off prefetch for every pending HF job up front (capped by PREFETCH).
for JOB in "${JOBS[@]}"; do
  IFS=$'\t' read -r IDX METHOD KIND NAME REPO_ID MODEL_DIR NEEDS_DL <<<"$JOB"
  is_done "$NAME" && continue
  [[ "$NEEDS_DL" == "1" ]] || continue
  start_prefetch "$NAME" "$REPO_ID" "$MODEL_DIR"
done

# ---- GPU workers: ready-first ----
declare -a FREE_GPUS=("${GPU_LIST[@]}")
declare -A PID_GPU=()
declare -A PID_LABEL=()
declare -A STARTED=()   # name -> 1 once assigned to a GPU

refresh_table() {
  python scripts/specgap_table.py --config "$CONFIG" --result-dir "$RESULT_DIR" \
    >/dev/null || echo "[warn] failed to refresh results/specgap_e0.md"
}

reap_audits() {
  local pid gpu refreshed=0
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      gpu="${PID_GPU[$pid]}"
      echo "[reap] GPU=$gpu finished ${PID_LABEL[$pid]:-job}"
      FREE_GPUS+=("$gpu")
      unset "PID_GPU[$pid]"
      unset "PID_LABEL[$pid]"
      refreshed=1
    fi
  done
  if [[ "$refreshed" == "1" ]]; then
    refresh_table
  fi
}

pending_left() {
  local job idx method kind name repo_id model_dir needs_dl
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r idx method kind name repo_id model_dir needs_dl <<<"$job"
    is_done "$name" && continue
    [[ -n "${STARTED[$name]:-}" ]] && continue
    return 0
  done
  return 1
}

pick_ready_job() {
  # Prefer jobs whose weights are already on disk.
  local job idx method kind name repo_id model_dir needs_dl
  local fallback=""
  for job in "${JOBS[@]}"; do
    IFS=$'\t' read -r idx method kind name repo_id model_dir needs_dl <<<"$job"
    is_done "$name" && continue
    [[ -n "${STARTED[$name]:-}" ]] && continue
    if is_ready "$model_dir"; then
      printf '%s' "$job"
      return 0
    fi
    # keep first not-ready as fallback only if nothing ready
    if [[ -z "$fallback" && "$needs_dl" == "1" ]]; then
      fallback="$job"
    fi
  done
  # Nothing ready: return empty (caller waits / keeps prefetching)
  return 1
}

echo "[pipeline] prefetch downloads in background; GPUs take ready checkpoints first"
refresh_table

while pending_left || [[ ${#PID_GPU[@]} -gt 0 ]]; do
  reap_downloads
  reap_audits

  # Keep prefetch queue filled for remaining jobs.
  for JOB in "${JOBS[@]}"; do
    IFS=$'\t' read -r IDX METHOD KIND NAME REPO_ID MODEL_DIR NEEDS_DL <<<"$JOB"
    is_done "$NAME" && continue
    [[ -n "${STARTED[$NAME]:-}" ]] && continue
    [[ "$NEEDS_DL" == "1" ]] || continue
    is_ready "$MODEL_DIR" && continue
    start_prefetch "$NAME" "$REPO_ID" "$MODEL_DIR"
  done

  # Assign free GPUs to ready jobs only.
  while [[ ${#FREE_GPUS[@]} -gt 0 ]]; do
    JOB="$(pick_ready_job || true)"
    if [[ -z "$JOB" ]]; then
      break
    fi
    IFS=$'\t' read -r IDX METHOD KIND NAME REPO_ID MODEL_DIR NEEDS_DL <<<"$JOB"
    GPU_ID="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    STARTED[$NAME]=1
    audit_ready "$IDX" "$METHOD" "$KIND" "$NAME" "$MODEL_DIR" "$GPU_ID" "$NEEDS_DL" &
    pid=$!
    PID_GPU[$pid]="$GPU_ID"
    PID_LABEL[$pid]="$METHOD/$KIND"
  done

  sleep 2
done

# Wait for any leftover prefetch processes.
for pid in "${!DL_PID[@]}"; do
  wait "$pid" 2>/dev/null || true
done

refresh_table
echo "[done] E0 batch finished (GPUs=$GPUS)"
cat "$ROOT/results/specgap_e0.md"
