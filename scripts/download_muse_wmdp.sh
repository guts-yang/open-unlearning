#!/bin/bash
# 入口：检查数据盘软链后转调 download_muse_wmdp_hf.sh。
# 不再走 S3 wget（慢且会写相对路径 data/wmdp；软链未建时会落到系统盘）。
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/download_muse_wmdp_hf.sh"
