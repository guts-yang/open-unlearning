#!/usr/bin/env python
"""Workspace train entry: register plugins, inject Hydra searchpath, guard output dir."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "workspace" / "src")]

from open_unlearning_plugins.paths import (  # noqa: E402
    assert_workspace_output_dir,
    hydra_searchpath_override,
    workspace_configs,
)
from open_unlearning_plugins.register import register_workspace_trainers  # noqa: E402

if not any(arg.startswith("hydra.searchpath") for arg in sys.argv[1:]):
    sys.argv.append(hydra_searchpath_override())

register_workspace_trainers()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from data import get_collators, get_data  # noqa: E402
from evals import get_evaluators  # noqa: E402
from model import get_model  # noqa: E402
from trainer import load_trainer  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


@hydra.main(
    version_base=None,
    config_path=str(REPO_ROOT / "configs"),
    config_name="unlearn.yaml",
)
def main(cfg: DictConfig):
    assert workspace_configs().is_dir(), workspace_configs()
    assert_workspace_output_dir(cfg)
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
