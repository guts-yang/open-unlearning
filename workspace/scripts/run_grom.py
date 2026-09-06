#!/usr/bin/env python
"""Thin aliases for GROM Hydra experiments. Same as workspace/scripts/train.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "tofu05": "unlearn/grom_tofu05",
    "tofu10": "unlearn/grom_tofu10",
    "muse_news": "unlearn/grom_muse_news",
    "muse_books": "unlearn/grom_muse_books",
    "wmdp_bio": "unlearn/grom_wmdp_bio",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        names = ", ".join(EXPERIMENTS)
        raise SystemExit(
            f"usage: python workspace/scripts/run_grom.py <{names}> [hydra overrides...]"
        )
    bench = sys.argv[1]
    if bench not in EXPERIMENTS:
        raise SystemExit(f"unknown bench {bench!r}; choose from {list(EXPERIMENTS)}")
    extra = sys.argv[2:]
    cmd = [
        sys.executable,
        str(REPO / "workspace" / "scripts" / "train.py"),
        f"experiment={EXPERIMENTS[bench]}",
        f"task_name=grom_{bench}",
        f"paths.output_dir=workspace/saves/unlearn/grom_{bench}",
        *extra,
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO)))


if __name__ == "__main__":
    main()
