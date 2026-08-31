"""Grid learning_rate × max_steps for frozen rank-1 MOGP-U spec (TOFU forget10).

Default is plan-only (no training). Pass --run after GPUs are available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SPEC = ROOT / "configs/mogpu/discovered/rank01_3d2389e5fe78.json"
RETAIN_LOGS = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
PRETRAINED = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
CACHED_50 = Path(
    "/root/autodl-tmp/saves/mogpu_open_search/search/g1_3d2389e5fe78_s0_n50/"
    "evals-50/TOFU_SUMMARY.json"
)
OUTPUT = Path("/root/autodl-tmp/saves/mogpu_rank01_grid")
LEARNING_RATES = (3e-6, 1e-5)
MAX_STEPS = (20, 50, 100)
MU_BASELINE = 0.53
FQ_ALTPO = 6.39e-6
FQ_COLLAPSE = 1e-12


def _port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _cell_key(lr: float, steps: int) -> tuple[float, int]:
    return (round(float(lr), 12), int(steps))


def _metrics(summary: Path) -> dict:
    values = json.loads(summary.read_text(encoding="utf-8"))
    rouge = float(values["forget_Q_A_ROUGE"])
    prob = float(values["forget_Q_A_Prob"])
    es = float(values["extraction_strength"])
    return {
        "forget_quality": float(values["forget_quality"]),
        "model_utility": float(values["model_utility"]),
        "forget_truth_ratio": float(values["forget_truth_ratio"]),
        "forget_Q_A_ROUGE": rouge,
        "forget_Q_A_Prob": prob,
        "extraction_strength": es,
        "privleak": float(values.get("privleak", float("nan"))),
        "one_minus_rouge": 1.0 - rouge,
        "one_minus_prob": 1.0 - prob,
        "one_minus_es": 1.0 - es,
    }


def rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda item: (
            -float(item["model_utility"]),
            -float(item["forget_quality"]),
        ),
    )


def recommend_10_epoch(rows: list[dict]) -> dict:
    """MU first; skip 10-epoch if utility is still low or FQ collapsed."""
    if not rows:
        return {"run_10_epoch": False, "reason": "grid is empty"}
    ranked = rank_rows(rows)
    best = ranked[0]
    mu = float(best["model_utility"])
    fq = float(best["forget_quality"])
    if mu < MU_BASELINE:
        return {
            "run_10_epoch": False,
            "reason": (
                f"best MU {mu:.4f} is still below local baseline {MU_BASELINE}; "
                "try larger RetainDrift weight or smaller kappa before 10 epoch"
            ),
            "best": best,
        }
    if fq < FQ_COLLAPSE:
        return {
            "run_10_epoch": False,
            "reason": f"best FQ {fq:.3e} collapsed; the cell barely unlearned",
            "best": best,
        }
    if fq <= FQ_ALTPO:
        return {
            "run_10_epoch": False,
            "reason": (
                f"best FQ {fq:.3e} is not clearly above local AltPO {FQ_ALTPO:.3e}"
            ),
            "best": best,
        }
    return {
        "run_10_epoch": True,
        "reason": (
            f"best MU {mu:.4f} reaches baseline and FQ {fq:.3e} stays above AltPO"
        ),
        "best": best,
    }


def _train(trial: Path, max_steps: int, lr: float) -> None:
    trial.mkdir(parents=True, exist_ok=True)
    command = [
        "accelerate",
        "launch",
        "--config_file",
        "configs/accelerate/default_config.yaml",
        "--main_process_port",
        str(_port()),
        "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/mogpu_search",
        "trainer=MOGPU",
        f"trainer.method_args.candidate_spec_path={SPEC}",
        f"task_name=rank01_lr{lr}_n{max_steps}",
        f"paths.output_dir={trial}",
        "trainer.args.seed=0",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.gradient_accumulation_steps=4",
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.gradient_checkpointing=true",
        "trainer.args.do_eval=false",
        "trainer.args.eval_strategy=no",
        f"trainer.args.learning_rate={lr}",
        f"trainer.args.max_steps={max_steps}",
        "trainer.args.save_strategy=steps",
        f"+trainer.args.save_steps={max_steps}",
        f"model.model_args.pretrained_model_name_or_path={PRETRAINED}",
        f"model.tokenizer_args.pretrained_model_name_or_path={PRETRAINED}",
        f"retain_logs_path={RETAIN_LOGS}",
        "forget_split=forget10",
        "retain_split=retain90",
        "holdout_split=holdout10",
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def _eval(model_path: Path, eval_dir: Path) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "src/eval.py",
        "experiment=eval/tofu/default",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
        f"paths.output_dir={eval_dir}",
        f"retain_logs_path={RETAIN_LOGS}",
        "forget_split=forget10",
        "holdout_split=holdout10",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    summary = eval_dir / "TOFU_SUMMARY.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="train and eval every cell; omit this flag to write the plan only",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        {
            "learning_rate": lr,
            "max_steps": steps,
            "reuse_open_search_f2": lr == 1e-5 and steps == 50,
        }
        for lr in LEARNING_RATES
        for steps in MAX_STEPS
    ]
    (output_dir / "grid_plan.json").write_text(
        json.dumps(
            {
                "spec": str(SPEC),
                "seed": 0,
                "cells": planned,
                "run": bool(args.run),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.run:
        preview = []
        if CACHED_50.is_file():
            preview.append(
                {
                    "learning_rate": 1e-5,
                    "max_steps": 50,
                    "seed": 0,
                    "reused_open_search_f2": True,
                    **_metrics(CACHED_50),
                }
            )
        gate = recommend_10_epoch(preview)
        (output_dir / "epoch10_recommendation.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "Wrote grid plan only (no training). Re-run with --run after GPUs are ready.",
            flush=True,
        )
        print(json.dumps(gate, indent=2), flush=True)
        return

    summary_path = output_dir / "grid_summary.json"
    rows = []
    if summary_path.is_file():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    done = {_cell_key(item["learning_rate"], item["max_steps"]) for item in rows}
    for lr in LEARNING_RATES:
        for steps in MAX_STEPS:
            key = _cell_key(lr, steps)
            if key in done:
                print(f"skip cached cell lr={lr} steps={steps}", flush=True)
                continue
            tag = f"lr{lr:g}_n{steps}"
            trial = output_dir / tag
            eval_dir = trial / f"evals-{steps}"
            summary = eval_dir / "TOFU_SUMMARY.json"
            reused = False
            if lr == 1e-5 and steps == 50 and CACHED_50.is_file():
                eval_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CACHED_50, summary)
                reused = True
                print(f"reuse open-search F2 for {tag}", flush=True)
            elif not (
                summary.is_file()
                and "model_utility" in json.loads(summary.read_text(encoding="utf-8"))
            ):
                if not (trial / "config.json").is_file():
                    _train(trial, steps, lr)
                summary = _eval(trial, eval_dir)
            metrics = _metrics(summary)
            row = {
                "spec": str(SPEC),
                "learning_rate": lr,
                "max_steps": steps,
                "seed": 0,
                "reused_open_search_f2": reused,
                "trial": str(trial),
                **metrics,
            }
            rows.append(row)
            summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row, indent=2), flush=True)
    ranked = rank_rows(rows)
    (output_dir / "grid_ranked.json").write_text(
        json.dumps(ranked, indent=2) + "\n", encoding="utf-8"
    )
    gate = recommend_10_epoch(rows)
    (output_dir / "epoch10_recommendation.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
