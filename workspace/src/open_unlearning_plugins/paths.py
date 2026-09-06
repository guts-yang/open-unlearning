from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    return repo_root() / "workspace"


def workspace_configs() -> Path:
    return workspace_root() / "configs"


def workspace_saves() -> Path:
    return workspace_root() / "saves"


def workspace_results() -> Path:
    return workspace_root() / "results"


def hydra_searchpath_override() -> str:
    return f"hydra.searchpath=[file://{workspace_configs()}]"


def assert_workspace_output_dir(cfg: DictConfig) -> Path:
    """Refuse training/eval dumps that would land in the official saves/ tree."""
    output = OmegaConf.select(cfg, "paths.output_dir")
    if output is None:
        raise SystemExit("paths.output_dir is missing; set it under workspace/saves/")
    output_path = Path(str(output)).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    else:
        output_path = output_path.resolve()
    allowed = workspace_saves().resolve()
    try:
        output_path.relative_to(allowed)
    except ValueError as exc:
        raise SystemExit(
            "paths.output_dir must be under workspace/saves/ "
            f"(got {output_path}; allowed prefix {allowed})"
        ) from exc
    os.makedirs(output_path, exist_ok=True)
    return output_path
