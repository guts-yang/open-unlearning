from __future__ import annotations

import pytest
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from open_unlearning_plugins.paths import (  # noqa: E402
    assert_workspace_output_dir,
    workspace_saves,
)


def test_guard_rejects_official_saves(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path / "saves" / "x")}})
    with pytest.raises(SystemExit):
        assert_workspace_output_dir(cfg)


def test_guard_accepts_workspace_saves(tmp_path):
    allowed = workspace_saves() / "unlearn" / "unit_test_guard"
    cfg = OmegaConf.create({"paths": {"output_dir": str(allowed)}})
    out = assert_workspace_output_dir(cfg)
    assert out == allowed.resolve()
    assert out.is_dir()
