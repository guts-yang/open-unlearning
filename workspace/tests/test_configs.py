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
        "configs/experiment/unlearn/blade_tofu_1b_01.yaml",
        "configs/experiment/unlearn/blade_muse_news.yaml",
        "configs/experiment/unlearn/baldro_npo_dv_tofu01.yaml",
        "docs/00_overview.md",
        "scripts/train.py",
    ):
        assert (root / rel).is_file(), rel
