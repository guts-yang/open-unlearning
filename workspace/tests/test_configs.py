from __future__ import annotations

from pathlib import Path


def test_workspace_plugin_configs_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "configs/trainer/LoRABiAL.yaml",
        "configs/trainer/LoRABiALAdaptive.yaml",
        "configs/trainer/DrNPO.yaml",
        "configs/trainer/GroupNPO.yaml",
        "configs/trainer/ALTER.yaml",
        "configs/trainer/GROM.yaml",
        "configs/experiment/unlearn/blade_tofu_1b_01.yaml",
        "configs/experiment/unlearn/grom_tofu10.yaml",
        "configs/experiment/unlearn/blade_muse_news.yaml",
        "configs/experiment/unlearn/baldro_npo_dv_tofu01.yaml",
        "docs/00_overview.md",
        "docs/05_eval_and_results.md",
        "docs/06_leakru_run.md",
        "scripts/train.py",
        "scripts/eval.py",
        "configs/eval/leakru.yaml",
        "configs/eval/tofu_and_leakru.yaml",
        "configs/eval/muse_and_leakru.yaml",
        "configs/eval/leakru_metrics/leakru_acc.yaml",
        "configs/eval/leakru_metrics/leakru_fq.yaml",
        "configs/eval/leakru_metrics/leakru_rr.yaml",
        "configs/leakru_datasets/leakru_mquake.yaml",
        "configs/experiment/eval/leakru_mquake.yaml",
        "configs/experiment/eval/tofu_leakru_mquake.yaml",
    ):
        assert (root / rel).is_file(), rel
