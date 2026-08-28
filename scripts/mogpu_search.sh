#!/bin/bash
# SAGE-Pareto + NSGA-II search (GPU fitness). Optional: dry_run=true
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p /root/autodl-tmp/saves/mogpu /root/autodl-tmp/logs
python scripts/run_mogpu_search.py "$@"
