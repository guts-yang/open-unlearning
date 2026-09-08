#!/usr/bin/env python
"""Workspace eval entry: register plugins, inject Hydra searchpath, guard output dir."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "workspace" / "src")]

from open_unlearning_plugins.paths import (  # noqa: E402
    assert_workspace_output_dir,
    hydra_searchpath_override,
)
from open_unlearning_plugins.register import (  # noqa: E402
    register_workspace_evaluators,
    register_workspace_trainers,
)

if not any(arg.startswith("hydra.searchpath") for arg in sys.argv[1:]):
    sys.argv.append(hydra_searchpath_override())

register_workspace_trainers()
register_workspace_evaluators()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from evals import get_evaluators  # noqa: E402
from model import get_model  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


@hydra.main(
    version_base=None,
    config_path=str(REPO_ROOT / "configs"),
    config_name="eval.yaml",
)
def main(cfg: DictConfig):
    assert_workspace_output_dir(cfg)
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for _evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
