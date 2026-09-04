#!/usr/bin/env python3
"""After the seed=0 grid: replicate the top-K hypers on seeds 1 and 2.

Ranking uses the same §F.2 score as search: HM(Mem, Utility) on seed=0
(or an untagged run, treated as seed 0). Priv / SpecGap are not used.
Existing aggregates are skipped.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from specdiff_table import discover_rows  # noqa: E402


SAVES = Path(os.environ.get("SAVES", "/root/autodl-tmp/saves"))
SPLIT = "forget10"
MODEL = "Llama-3.2-1B-Instruct"


def seed_of(row):
    return 0 if row["seed"] is None else int(row["seed"])


def seed0_score(group):
    seed0 = [row for row in group if seed_of(row) == 0]
    if not seed0:
        return None
    row = max(seed0, key=lambda item: item["HM_MU"])
    return row["HM_MU"], row


def replica_tag(sample_tag, seed):
    base = re.sub(r"_seed\d+$", "", sample_tag)
    return f"{base}_seed{seed}"


def aggregate_exists(tag):
    return (
        SAVES / "unlearn" / f"tofu_1B_SpecDiff_{SPLIT}_{tag}" / "evals" / "ou_aggregate.json"
    ).is_file()


def launch(row, tag, seed):
    cmd = [
        "bash",
        "scripts/tofu_unlearn_one.sh",
        "SpecDiff",
        SPLIT,
        MODEL,
        tag,
        f"trainer.args.learning_rate={row['lr']}",
        "trainer.args.optim=adamw_torch",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.method_args.draft_model_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        f"trainer.method_args.lam={row['lam']}",
        f"trainer.method_args.beta={row['beta']}",
        f"trainer.method_args.kappa={row['kappa']}",
        f"trainer.method_args.tau={row['tau']}",
        "trainer.method_args.warmup_steps=1",
        "trainer.method_args.chunk_size=8192",
        f"trainer.args.seed={seed}",
        "trainer.args.logging_steps=5",
        f"trainer.args.num_train_epochs={int(row['epochs'])}",
    ]
    print(f"[replicate] hyper={row['hyper']} seed={seed} tag={tag}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = defaultdict(list)
    for row in discover_rows():
        if row["tag"] in {"smoke_s0"}:
            continue
        groups[row["hyper"]].append(row)

    ranked = []
    for hyper, group in groups.items():
        scored = seed0_score(group)
        if scored is None:
            print(f"[skip] no seed=0 for {hyper}", flush=True)
            continue
        hm, sample = scored
        ranked.append((hm, hyper, group, sample))
    ranked.sort(reverse=True)

    chosen = ranked[: args.top]
    print(
        f"[replicate] ranking {len(ranked)} hypers by seed0 HM; taking top {len(chosen)}",
        flush=True,
    )
    for rank, (hm, hyper, group, sample) in enumerate(chosen, start=1):
        have = sorted({seed_of(row) for row in group})
        print(f"  #{rank} HM={hm:.4f} {hyper} seeds={have}", flush=True)

    jobs = []
    for hm, hyper, group, sample in chosen:
        for seed in args.seeds:
            tag = replica_tag(sample["tag"], seed)
            if aggregate_exists(tag) or any(seed_of(row) == seed for row in group):
                print(f"[skip] exists {hyper} seed={seed} tag={tag}", flush=True)
                continue
            jobs.append((sample, tag, seed))

    if args.dry_run:
        for sample, tag, seed in jobs:
            print(f"[dry-run] would run {sample['hyper']} seed={seed} tag={tag}")
        return 0

    failures = 0
    for sample, tag, seed in jobs:
        rc = launch(sample, tag, seed)
        if rc != 0:
            print(f"[fail] tag={tag} rc={rc}", flush=True)
            failures += 1

    table = subprocess.run(["python", "scripts/specdiff_table.py"], cwd=ROOT)
    if table.returncode != 0:
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
