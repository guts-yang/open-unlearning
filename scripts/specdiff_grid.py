#!/usr/bin/env python3
"""Expand and optionally launch the SpecDiff TOFU-1B search grid."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "scripts" / "specdiff_grid.yaml"
SAVES = Path(os.environ.get("SAVES", "/root/autodl-tmp/saves"))


def load_points():
    spec = yaml.safe_load(GRID_PATH.read_text())
    points = []
    for stage in ("stage_a", "stage_b", "stage_c"):
        for raw in spec[stage]:
            point = dict(raw)
            point["stage"] = stage
            point["tag"] = tag_for(point)
            points.append(point)
    return spec, points


def tag_for(point):
    lr = point["lr"]
    lr_s = f"{lr:.0e}".replace("+0", "").replace("-0", "-")
    return (
        f"g_lr{lr_s}_lam{point['lam']:g}_b{point['beta']:g}"
        f"_k{point['kappa']:g}_ep{int(point['epochs'])}"
    )


def aggregate_path(tag):
    return SAVES / "unlearn" / f"tofu_1B_SpecDiff_forget10_{tag}" / "evals" / "ou_aggregate.json"


def fmt_row(point):
    status = "done" if point.get("done") or aggregate_path(point["tag"]).is_file() else "todo"
    return (
        f"| {point['stage']} | `{point['tag']}` | {point['lr']:g} | "
        f"{int(point['epochs'])} | {point['lam']:g} | {point['beta']:g} | "
        f"{point['kappa']:g} | {status} |"
    )


def launch(point):
    env = os.environ.copy()
    env.update(
        {
            "SPECDIFF_TAG": point["tag"].removesuffix(f"_ep{int(point['epochs'])}"),
            "SPECDIFF_LAM": str(point["lam"]),
            "SPECDIFF_BETA": str(point["beta"]),
            "SPECDIFF_KAPPA": str(point["kappa"]),
            "SPECDIFF_TAU": str(point["tau"]),
        }
    )
    # Tags from specdiff_tofu.sh are ${SPECDIFF_TAG}_seed${seed}. Use a dedicated runner.
    cmd = [
        "bash",
        "scripts/tofu_unlearn_one.sh",
        "SpecDiff",
        "forget10",
        "Llama-3.2-1B-Instruct",
        point["tag"],
        "trainer.args.learning_rate=" + str(point["lr"]),
        "trainer.args.optim=adamw_torch",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.method_args.draft_model_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        f"trainer.method_args.lam={point['lam']}",
        f"trainer.method_args.beta={point['beta']}",
        f"trainer.method_args.kappa={point['kappa']}",
        f"trainer.method_args.tau={point['tau']}",
        "trainer.method_args.warmup_steps=1",
        "trainer.method_args.chunk_size=8192",
        "trainer.args.seed=0",
        "trainer.args.logging_steps=5",
        f"trainer.args.num_train_epochs={int(point['epochs'])}",
    ]
    print("[run]", " ".join(cmd[3:8]), point["tag"])
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="train+eval remaining seed=0 points")
    args = parser.parse_args()
    spec, points = load_points()
    todo = [
        p
        for p in points
        if not p.get("done") and not aggregate_path(p["tag"]).is_file()
    ]
    print(f"grid points={len(points)} todo={len(todo)} skip_done={len(points) - len(todo)}")
    if not args.run:
        for point in points:
            print(fmt_row(point))
        return
    for point in todo:
        launch(point)
    subprocess.run(["python", "scripts/specdiff_table.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
