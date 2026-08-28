"""Run SAGE-Pareto + NSGA-II search using frozen MOGPU training."""

from __future__ import annotations

import json
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
    output_dir = Path(cfg.output_dir)
    recipe = {
        **OmegaConf.to_container(cfg.fixed_recipe, resolve=True),
        "repo_root": str(cfg.repo_root),
        "output_dir": str(output_dir),
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "protocol_dir": str(Path(cfg.repo_root) / "configs/mogpu/search"),
        "retain_logs_path": str(cfg.retain_logs_path),
        "fq_threshold": float(cfg.fq_threshold),
        "training_seed": 0,
        "forget_split": "forget10",
        "cache_policy": "reuse_succeeded_only",
        "dry_run": bool(cfg.get("dry_run", False)),
        "budget": OmegaConf.to_container(cfg.fixed_recipe.budget, resolve=True),
        "nsga_overrides": OmegaConf.to_container(
            cfg.get("nsga_overrides") or {}, resolve=True
        ),
        "skip_f3": bool(cfg.get("skip_f3", False)),
    }
    controller = SearchController(registry, output_dir, recipe)
    result = controller.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
