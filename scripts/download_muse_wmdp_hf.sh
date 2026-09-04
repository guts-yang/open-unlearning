#!/bin/bash
# MUSE / WMDP 预下载（HF 镜像优先，失败则 ModelScope；全部落数据盘）。
# 用法：
#   source /root/autodl-tmp/env_hf.sh && source /root/autodl-tmp/envs/unlearning/bin/activate
#   bash scripts/download_muse_wmdp_hf.sh
# 幂等：已 [OK] 的项跳过；失败可重跑续传。
# 状态：/root/autodl-tmp/logs/download_muse_wmdp_hf.status
set -uo pipefail

source /root/autodl-tmp/env_hf.sh
# shellcheck disable=SC1091
source /root/autodl-tmp/envs/unlearning/bin/activate
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope}"
mkdir -p "$MODELSCOPE_CACHE"

# hf-mirror.com 在本机经常 TCP 通、TLS 握手挂死；hf-mirror.net 可用时切过去。
pick_hf_endpoint () {
  local ep
  for ep in "${HF_ENDPOINT:-https://hf-mirror.com}" https://hf-mirror.net https://hf-mirror.com; do
    if timeout 8 curl -sfI --max-time 6 "$ep/" >/dev/null 2>&1; then
      export HF_ENDPOINT="$ep"
      echo "[info] HF_ENDPOINT=$HF_ENDPOINT"
      return 0
    fi
  done
  echo "[warn] 无可用 HF 镜像，将主要依赖 ModelScope"
}
pick_hf_endpoint

ROOT=/usr/local/open-unlearning
DATA=/root/autodl-tmp
LOG_DIR="$DATA/logs"
STATUS="$LOG_DIR/download_muse_wmdp_hf.status"
HUB="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$LOG_DIR" "$HUB"
cd "$ROOT"

assert_on_data_disk () {
  local label="$1" path="$2"
  local real
  real="$(readlink -f "$path" 2>/dev/null || true)"
  if [[ -z "$real" || "$real" != /root/autodl-tmp* ]]; then
    echo "[error] $label 必须在 /root/autodl-tmp 下，当前: path=$path real=${real:-missing}"
    exit 1
  fi
}

assert_on_data_disk HF_HOME "${HF_HOME:?source env_hf.sh first}"
assert_on_data_disk HUGGINGFACE_HUB_CACHE "$HUB"
assert_on_data_disk saves "$ROOT/saves"
assert_on_data_disk "data/wmdp" "$ROOT/data/wmdp"
assert_on_data_disk MODELSCOPE_CACHE "$MODELSCOPE_CACHE"

mark () { echo "$(date '+%F %T') [$1] $2" | tee -a "$STATUS"; }
done_ok () { grep -q "\[OK\] $1\$" "$STATUS" 2>/dev/null; }

# hf-mirror 会偶发 ConnectTimeout / TLS hang；hub 支持断点续传，失败再走 ModelScope。
retry () {
  local tries="${1:-5}" delay=15
  shift
  local i
  for ((i=1; i<=tries; i++)); do
    if "$@"; then
      return 0
    fi
    echo "[retry $i/$tries] $*"
    sleep "$delay"
    delay=$((delay * 2))
    if (( delay > 120 )); then delay=120; fi
  done
  return 1
}

hub_dir () {
  local repo_type="$1" repo="$2"
  local prefix
  if [[ "$repo_type" == dataset ]]; then prefix=datasets; else prefix=models; fi
  echo "$HUB/${prefix}--${repo//\//--}"
}

plant_snap () {
  local repo_type="$1" repo="$2"
  local dest snap
  dest="$(hub_dir "$repo_type" "$repo")"
  snap="$dest/snapshots/modelscope"
  mkdir -p "$dest/refs" "$snap"
  printf 'modelscope\n' > "$dest/refs/main"
  echo "$snap"
}

# 权重或数据集已在 HF hub 布局里则跳过下载。
hub_ready () {
  local repo_type="$1" repo="$2"
  python - "$repo_type" "$repo" "$HUB" <<'PY'
import sys
from pathlib import Path
repo_type, repo, hub = sys.argv[1], sys.argv[2], Path(sys.argv[3])
prefix = "datasets" if repo_type == "dataset" else "models"
root = hub / f"{prefix}--{repo.replace('/', '--')}"
if not root.is_dir():
    raise SystemExit(1)
snaps = list((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
cands = snaps + [root]
need = ("*.safetensors", "*.bin", "*.parquet", "*.arrow", "*.jsonl", "*.json")
for c in cands:
    if not c.is_dir():
        continue
    for pat in need:
        hits = [p for p in c.rglob(pat[1:]) if p.is_file() and p.stat().st_size > 0]
        # rglob('*.json') via suffix
        hits = [p for p in c.rglob("*") if p.is_file() and p.stat().st_size > 64
                and p.suffix in {".safetensors", ".bin", ".parquet", ".arrow", ".jsonl"}]
        if hits:
            raise SystemExit(0)
        # small datasets: README + dataset script / json configs
        if repo_type == "dataset" and (c / "README.md").is_file():
            extras = [p for p in c.rglob("*") if p.is_file() and p.suffix in {".py", ".yml", ".yaml", ".json", ".md"}]
            if len(extras) >= 2:
                raise SystemExit(0)
raise SystemExit(1)
PY
}

ms_snapshot () {
  local repo_type="$1" repo="$2" dest="$3"
  python - "$repo_type" "$repo" "$dest" "$MODELSCOPE_CACHE" <<'PY'
import os, sys
from pathlib import Path
repo_type, repo, dest, cache = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.environ.setdefault("MODELSCOPE_CACHE", cache)
Path(dest).mkdir(parents=True, exist_ok=True)
from modelscope import snapshot_download
snapshot_download(
    repo,
    repo_type=repo_type,
    local_dir=dest,
    cache_dir=cache,
    ignore_file_pattern=["*.bin"] if repo_type == "model" else None,
)
print("modelscope ok", repo, dest)
PY
}

dl_repo () {
  local repo_type="$1" repo="$2" key="$3"
  shift 3
  if hub_ready "$repo_type" "$repo"; then
    echo "[hub-ready] $repo"
    return 0
  fi
  echo "[get] $repo via $HF_ENDPOINT (no-HEAD)"
  if retry 3 python "$ROOT/scripts/_hf_get_snapshot.py" "$repo" "$repo_type"; then
    return 0
  fi
  if retry 2 huggingface-cli download "$repo" --repo-type "$repo_type"; then
    return 0
  fi
  echo "[fallback] ModelScope $repo"
  local snap
  snap="$(plant_snap "$repo_type" "$repo")"
  ms_snapshot "$repo_type" "$repo" "$snap"
}

# --- MUSE 数据集 -----------------------------------------------------------------
for ds in MUSE-News MUSE-Books; do
  done_ok "dataset/$ds" && { mark SKIP "dataset/$ds"; continue; }
  mark RUN "dataset/$ds"
  if dl_repo dataset "muse-bench/$ds" "dataset/$ds" >>"$LOG_DIR/dl_$ds.log" 2>&1; then
    mark OK "dataset/$ds"
  else
    mark FAIL "dataset/$ds"
  fi
done

# --- MUSE target 7B -------------------------------------------------------------
for m in MUSE-News_target MUSE-Books_target; do
  done_ok "model/$m" && { mark SKIP "model/$m"; continue; }
  mark RUN "model/$m"
  if dl_repo model "muse-bench/$m" "model/$m" >>"$LOG_DIR/dl_$m.log" 2>&1; then
    mark OK "model/$m"
  else
    mark FAIL "model/$m"
  fi
done

# --- zephyr-7b-beta -------------------------------------------------------------
if done_ok model/zephyr; then
  mark SKIP model/zephyr
else
  mark RUN model/zephyr
  if dl_repo model HuggingFaceH4/zephyr-7b-beta model/zephyr \
       >>"$LOG_DIR/dl_zephyr.log" 2>&1; then
    mark OK model/zephyr
  else
    mark FAIL model/zephyr
  fi
fi

# --- cais/wmdp MCQ --------------------------------------------------------------
if done_ok wmdp-mcq; then
  mark SKIP wmdp-mcq
else
  mark RUN wmdp-mcq
  if dl_repo dataset cais/wmdp wmdp-mcq >>"$LOG_DIR/dl_wmdp_mcq.log" 2>&1 \
     && python - >>"$LOG_DIR/dl_wmdp_mcq.log" 2>&1 <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
local = hub / "datasets--cais--wmdp" / "snapshots" / "modelscope"
names = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
ok = False
if local.is_dir():
    try:
        ds = load_dataset(str(local))
        print("load local keys", list(ds.keys()) if hasattr(ds, "keys") else ds)
        ok = True
    except Exception as e:
        print("load local failed", e)
if not ok:
    for name in names:
        try:
            load_dataset("cais/wmdp", name)
            print("ok hf", name)
            ok = True
        except Exception as e:
            print("hf", name, e)
if not ok:
    from modelscope.msdatasets import MsDataset
    for name in names:
        MsDataset.load("cais/wmdp", subset_name=name)
        print("ok ms", name)
print("wmdp mcq ready")
PY
  then
    mark OK wmdp-mcq
  else
    mark FAIL wmdp-mcq
  fi
fi

# --- mmlu（WMDP utility） -------------------------------------------------------
if done_ok dataset/mmlu; then
  mark SKIP dataset/mmlu
else
  mark RUN dataset/mmlu
  if { dl_repo dataset cais/mmlu dataset/mmlu \
       || dl_repo dataset cais/mmlu dataset/mmlu; } >>"$LOG_DIR/dl_mmlu.log" 2>&1 \
     && python - >>"$LOG_DIR/dl_mmlu.log" 2>&1 <<'PY'
import os
from pathlib import Path
from datasets import load_dataset
hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
local = hub / "datasets--cais--mmlu" / "snapshots" / "modelscope"
ok = False
if local.is_dir():
    try:
        load_dataset(str(local), "abstract_algebra")
        ok = True
        print("ok local mmlu")
    except Exception as e:
        print("local mmlu", e)
        try:
            load_dataset(str(local))
            ok = True
        except Exception as e2:
            print("local mmlu default", e2)
if not ok:
    try:
        load_dataset("cais/mmlu", "all")
        ok = True
    except Exception:
        try:
            load_dataset("cais/mmlu", "abstract_algebra")
            ok = True
        except Exception as e:
            print("hf mmlu", e)
if not ok:
    from modelscope.msdatasets import MsDataset
    MsDataset.load("cais/mmlu", subset_name="abstract_algebra")
    print("ok ms mmlu")
    ok = True
if not ok:
    raise SystemExit("mmlu failed")
print("ok mmlu")
PY
  then
    mark OK dataset/mmlu
  else
    mark FAIL dataset/mmlu
  fi
fi

# --- MUSE 官方 retrain 对照日志（不要全量 56 个 eval 文件） ----------------------
if done_ok muse-eval-logs; then
  mark SKIP muse-eval-logs
elif [[ -f "$DATA/saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json" \
     && -f "$DATA/saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json" ]]; then
  mark OK muse-eval-logs
else
  mark RUN muse-eval-logs
  if retry 2 huggingface-cli download open-unlearning/eval --repo-type dataset \
       --include "muse_*" \
       --local-dir "$DATA/saves/eval" \
       >>"$LOG_DIR/dl_muse_eval_logs.log" 2>&1 \
     || python - >>"$LOG_DIR/dl_muse_eval_logs.log" 2>&1 <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download
dest = Path("/root/autodl-tmp/saves/eval")
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(
    "open-unlearning/eval",
    repo_type="dataset",
    local_dir=str(dest),
    allow_file_pattern=["muse_*"],
    cache_dir=os.environ.get("MODELSCOPE_CACHE", "/root/autodl-tmp/modelscope"),
)
need = [
    dest / "muse_Llama-2-7b-hf_News_retrain" / "MUSE_EVAL.json",
    dest / "muse_Llama-2-7b-hf_Books_retrain" / "MUSE_EVAL.json",
]
missing = [str(p) for p in need if not p.is_file()]
if missing:
    raise SystemExit("missing " + ", ".join(missing))
print("muse eval logs ready")
PY
  then
    mark OK muse-eval-logs
  else
    mark FAIL muse-eval-logs
  fi
fi

# --- WMDP corpus：cais/wmdp-corpora parquet → jsonl（不走 S3） --------------------
if done_ok wmdp-corpus; then
  mark SKIP wmdp-corpus
else
  mark RUN wmdp-corpus
  if python - >>"$LOG_DIR/dl_wmdp_corpora.log" 2>&1 <<'PY'
import json
import os
from pathlib import Path

root = Path("/root/autodl-tmp/data/wmdp")
hf_dir = root / "wmdp-corpora_hf"
jsonl_dir = root / "wmdp-corpora_jsonl"
hf_dir.mkdir(parents=True, exist_ok=True)
jsonl_dir.mkdir(parents=True, exist_ok=True)

have_pq = any(hf_dir.rglob("*.parquet"))
if not have_pq:
    os.environ.setdefault("MODELSCOPE_CACHE", "/root/autodl-tmp/modelscope")
    try:
        from huggingface_hub import snapshot_download as hf_snap
        hf_snap("cais/wmdp-corpora", repo_type="dataset", local_dir=str(hf_dir))
    except Exception as e:
        print("hf snapshot failed", e)
        from modelscope import snapshot_download
        snapshot_download(
            "cais/wmdp-corpora",
            repo_type="dataset",
            local_dir=str(hf_dir),
            cache_dir=os.environ["MODELSCOPE_CACHE"],
        )

def write_jsonl(name, rows):
    out = jsonl_dir / f"{name}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            if "text" not in row:
                # keep first string-ish column as text
                for k, v in row.items():
                    if isinstance(v, str) and k != "text":
                        row = {"text": v, **{kk: vv for kk, vv in row.items() if kk != k}}
                        break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {out} n={n} bytes={out.stat().st_size}")

copied = False
for p in list(hf_dir.rglob("*.jsonl")):
    dest = jsonl_dir / p.name
    if not dest.exists():
        dest.write_bytes(p.read_bytes())
    print("copied jsonl", p.name)
    copied = True

if not copied:
    from datasets import load_dataset
    groups = {}
    for pq in sorted(hf_dir.rglob("*.parquet")):
        # .../cyber-forget-corpus/train-00000-of-00001.parquet
        name = pq.parent.name if pq.parent != hf_dir else pq.stem
        groups.setdefault(name, []).append(str(pq))
    print("parquet groups", {k: len(v) for k, v in groups.items()})
    for name, files in groups.items():
        table = load_dataset("parquet", data_files=files, split="train")
        write_jsonl(name, [dict(r) for r in table])

link = root / "wmdp-corpora"
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to("wmdp-corpora_jsonl")

need = [
    jsonl_dir / "cyber-forget-corpus.jsonl",
    jsonl_dir / "cyber-retain-corpus.jsonl",
]
missing = [str(p) for p in need if not p.exists() or p.stat().st_size == 0]
if missing:
    raise SystemExit("missing required jsonl: " + ", ".join(missing))
print("wmdp corpus ready")
PY
  then
    mark OK wmdp-corpus
  else
    mark FAIL wmdp-corpus
  fi
fi

mark DONE "MUSE/WMDP HF 下载项执行完毕"
echo "---- 磁盘 ----"
df -h / /root/autodl-tmp | cat
echo "---- 软链 ----"
readlink -f "$ROOT/saves" "$ROOT/data/wmdp" "$ROOT/logs"
