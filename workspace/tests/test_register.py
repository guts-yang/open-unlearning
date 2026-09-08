from __future__ import annotations

import pytest

pytest.importorskip("torch")

from open_unlearning_plugins.register import (  # noqa: E402
    register_workspace_evaluators,
    register_workspace_trainers,
)
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
        "GROM",
    ):
        assert name in names
        assert name in TRAINER_REGISTRY
    ev = register_workspace_evaluators()
    assert "LeakRUEvaluator" in ev
