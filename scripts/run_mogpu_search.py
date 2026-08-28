"""Run a sequential MOGP-U seed search using the repository's Hydra training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mogpu.search import SearchController
from mogpu.seed_registry import SeedRegistry


@hydra.main(version_base=None, config_path="../configs/mogpu", config_name="default")
def main(cfg: DictConfig) -> None:
    registry = SeedRegistry.load(cfg.seed_catalog_path)
    recipe = {
        **OmegaConf.to_container(cfg.fixed_recipe, resolve=True),
        "repo_root": cfg.repo_root,
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "model": "configured_by_experiment",
        "tofu_split": "configured_by_experiment",
        "training_seed": 0,
        "budget": OmegaConf.to_container(cfg.fixed_recipe.budget, resolve=True),
    }
    controller = SearchController(registry, cfg.output_dir, recipe)
    for seed in registry.initial_population(benchmark="TOFU"):
        controller.run_seed(seed.seed_id)


if __name__ == "__main__":
    main()
