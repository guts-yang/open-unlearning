from __future__ import annotations

import pytest

pytest.importorskip("torch")

from open_unlearning_plugins.register import register_workspace_trainers  # noqa: E402
from trainer import TRAINER_REGISTRY  # noqa: E402


def test_plugin_handlers_register_without_hf_download():
    names = register_workspace_trainers()
    for name in (
        "LoRABiALAdaptive",
        "DrNPO",
        "GroupNPO",
        "DrSimNPO",
        "GroupSimNPO",
        "DrSatImp",
        "GroupSatImp",
        "ALTER",
    ):
        assert name in names
        assert name in TRAINER_REGISTRY
