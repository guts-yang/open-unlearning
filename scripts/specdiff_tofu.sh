#!/bin/bash
# SpecDiff TOFU-1B smoke and three-seed experiment.
#
#   bash scripts/specdiff_tofu.sh smoke
#   bash scripts/specdiff_tofu.sh seeds
#   bash scripts/specdiff_tofu.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
MODEL="Llama-3.2-1B-Instruct"
SPLIT="forget10"
SAVES="${SAVES:-/root/autodl-tmp/saves}"
TAG_PREFIX="${SPECDIFF_TAG:+${SPECDIFF_TAG}_}"
LAM="${SPECDIFF_LAM:-1.0}"
BETA="${SPECDIFF_BETA:-0.1}"
KAPPA="${SPECDIFF_KAPPA:-0.3}"
TAU="${SPECDIFF_TAU:-0.02}"
COMMON_ARGS=(
  trainer.args.learning_rate=5e-5
  trainer.args.optim=adamw_torch
  trainer.args.do_eval=false
  trainer.args.eval_on_start=false
  trainer.args.eval_strategy=no
  trainer.method_args.draft_model_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
  trainer.method_args.lam="$LAM"
  trainer.method_args.beta="$BETA"
  trainer.method_args.kappa="$KAPPA"
  trainer.method_args.tau="$TAU"
  trainer.method_args.warmup_steps=1
  trainer.method_args.chunk_size=8192
)

run_one() {
  local tag="$1"
  shift
  local aggregate="$SAVES/unlearn/tofu_1B_SpecDiff_${SPLIT}_${tag}/evals/ou_aggregate.json"
  if [[ -f "$aggregate" ]]; then
    echo "[skip] completed SpecDiff $tag"
    return
  fi
  bash scripts/tofu_unlearn_one.sh SpecDiff "$SPLIT" "$MODEL" "$tag" \
    "${COMMON_ARGS[@]}" "$@"
}

run_smoke() {
  run_one smoke_s0 \
    trainer.args.seed=0 \
    trainer.args.logging_steps=1 \
    +trainer.args.max_steps=5 \
    trainer.args.num_train_epochs=1
  python - "$SAVES/unlearn/tofu_1B_SpecDiff_${SPLIT}_smoke_s0/evals/ou_aggregate.json" <<'PY'
import json
import math
import sys

result = json.load(open(sys.argv[1]))
required = ("Mem", "Priv", "Utility", "Agg", "SpecGap")
missing = [key for key in required if key not in result]
if missing:
    raise SystemExit(f"smoke output missing: {missing}")
values = [
    result["Mem"],
    result["Priv"],
    result["Utility"],
    result["Agg"],
    result["SpecGap"]["forget"]["mean"],
    result["SpecGap"]["retain"]["mean"],
]
if not all(math.isfinite(float(value)) for value in values):
    raise SystemExit("smoke output contains non-finite metrics")
print("[gate] SpecDiff smoke passed")
PY
}

run_seeds() {
  local seed
  for seed in 0 1 2; do
    run_one "${TAG_PREFIX}seed${seed}" \
      trainer.args.seed="$seed" \
      trainer.args.logging_steps=5 \
      trainer.args.num_train_epochs=10
  done
  python scripts/specdiff_table.py
}

ALT_JSON="${ALT_JSON:-$ROOT/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget10/alt5_seed_0.json}"

alt_data_args() {
  if [[ ! -f "$ALT_JSON" ]]; then
    echo "[error] missing AltPO jsonl: $ALT_JSON" >&2
    return 1
  fi
  printf '%s\n' \
    "data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset" \
    "~data.forget.TOFU_QA_forget.args.hf_args.name" \
    "data.forget.TOFU_QA_forget.args.hf_args.path=json" \
    "+data.forget.TOFU_QA_forget.args.hf_args.data_files=$ALT_JSON" \
    "data.forget.TOFU_QA_forget.args.hf_args.split=train" \
    "+data.forget.TOFU_QA_forget.args.alternate_key=alternate" \
    "+data.forget.TOFU_QA_forget.args.return_original=True"
}

run_warmup_only() {
  # One GradDiff optimizer step, then full eval. Locates Utility damage.
  run_one warmup_only_s0 \
    trainer.args.seed=0 \
    trainer.args.logging_steps=1 \
    trainer.args.num_train_epochs=1 \
    +trainer.args.max_steps=1 \
    trainer.method_args.warmup_steps=1 \
    trainer.method_args.warmup_kind=graddiff
  python - "$SAVES/unlearn/tofu_1B_SpecDiff_${SPLIT}_warmup_only_s0/evals/ou_aggregate.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
print(
    "[warmup-only]",
    f"Mem={result['Mem']:.4f}",
    f"Priv={result['Priv']:.4f}",
    f"Utility={result['Utility']:.4f}",
    f"Agg={result['Agg']:.4f}",
    f"SpecGap_f={result['SpecGap']['forget']['mean']:.4f}",
    f"SpecGap_r={result['SpecGap']['retain']['mean']:.4f}",
)
if result["Utility"] < 0.85:
    print("[read] Utility already low after 1 GradDiff step → damage is in warmup")
else:
    print("[read] Utility still high after 1 GradDiff step → damage is in the SpecDiff loop")
PY
}

run_lr2e5_seeds() {
  local seed
  for seed in 1 2; do
    run_one "g_lr2e-5_lam1_b0.1_k0.3_ep10_seed${seed}" \
      trainer.args.seed="$seed" \
      trainer.args.logging_steps=5 \
      trainer.args.learning_rate=2e-5 \
      trainer.args.num_train_epochs=10 \
      trainer.method_args.lam=1.0 \
      trainer.method_args.beta=0.1 \
      trainer.method_args.kappa=0.3
  done
  python scripts/specdiff_table.py
}

run_generate_alt() {
  mkdir -p "$(dirname "$ALT_JSON")"
  if [[ -f "$ALT_JSON" ]]; then
    echo "[skip] $ALT_JSON exists"
    return 0
  fi
  (
    cd "$ROOT/community/methods/AltPO"
    python generate.py \
      dataset_config.dataset_kwargs.name=forget10 \
      output_file="$ALT_JSON"
  )
  python - "$ALT_JSON" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
changed = False
for row in rows:
    alt = row.get("alternate")
    if isinstance(alt, list):
        row["alternate"] = alt[0] if alt else ""
        changed = True
if changed:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
print(f"[alt] n={len(rows)} file={path} coerced_list={changed}")
if len(rows) < 400:
    raise SystemExit(f"expected >=400 forget10 rows (M=5 may duplicate Qs), got {len(rows)}")
empty = sum(1 for row in rows if not str(row.get("alternate") or "").strip())
print(f"[alt] empty_alternate={empty}")
print("[alt] sample 20:")
seen_q = []
for row in rows:
    q = row["question"]
    if q in seen_q:
        continue
    seen_q.append(q)
    print(" Q:", q[:100])
    print(" A:", str(row["answer"])[:100])
    print(" Y:", str(row["alternate"])[:160])
    print("---")
    if len(seen_q) >= 20:
        break
PY
}

run_v1() {
  mapfile -t extra < <(alt_data_args)
  run_one altspec_v1_wudpo2_seed0 \
    trainer.args.seed=0 \
    trainer.args.logging_steps=5 \
    trainer.args.num_train_epochs=10 \
    trainer.method_args.warmup_steps=2 \
    trainer.method_args.warmup_kind=dpo \
    trainer.method_args.dpo_weight=0.0 \
    trainer.method_args.dpo_beta=0.1 \
    "${extra[@]}"
  python scripts/specdiff_table.py
}

run_v2() {
  local lr tag
  mapfile -t extra < <(alt_data_args)
  for lr in 5e-5 2e-5; do
    tag="altspec_v2_dpo0.5_lr${lr}_seed0"
    run_one "$tag" \
      trainer.args.seed=0 \
      trainer.args.logging_steps=5 \
      trainer.args.learning_rate="$lr" \
      trainer.args.num_train_epochs=10 \
      trainer.method_args.warmup_steps=1 \
      trainer.method_args.warmup_kind=graddiff \
      trainer.method_args.dpo_weight=0.5 \
      trainer.method_args.dpo_beta=0.1 \
      trainer.method_args.kappa=0.3 \
      "${extra[@]}"
  done
  python scripts/specdiff_table.py
}

case "$MODE" in
  smoke) run_smoke ;;
  seeds) run_seeds ;;
  warmup) run_warmup_only ;;
  lr2e5) run_lr2e5_seeds ;;
  generate) run_generate_alt ;;
  v1) run_v1 ;;
  v2) run_v2 ;;
  all)
    run_smoke
    run_seeds
    ;;
  grid)
    python scripts/specdiff_grid.py
    ;;
  *) echo "usage: $0 [smoke|seeds|all|grid|warmup|lr2e5|generate|v1|v2]"; exit 2 ;;
esac
