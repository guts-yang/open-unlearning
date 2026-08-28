"""Sequential, fixed-recipe MOGP-U search controller."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from mogpu.ledger import Ledger
from mogpu.seed_registry import SeedRegistry
from mogpu.seeds import write_candidate
from trainer.unlearn.mogpu_dsl.render import render_card


class SearchController:
    def __init__(
        self, registry: SeedRegistry, output_dir: str | Path, recipe: dict[str, Any]
    ):
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.recipe = recipe
        self._validate_recipe()

    def _validate_recipe(self) -> None:
        forbidden = {
            "learning_rate",
            "steps",
            "batch_size",
            "data_order",
            "model",
            "evaluator",
        }
        if forbidden & set(self.recipe.get("searchable", [])):
            raise ValueError("Training recipe fields cannot be GP genes")
        if self.recipe.get("lora_rank") is not None:
            raise ValueError("MOGP-U v2 requires lora_rank: null")

    @property
    def config_hash(self) -> str:
        return sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest()

    def run_seed(
        self, seed_id: str, generation: int = 0, parents: list[str] | None = None
    ):
        seed = next(
            item
            for item in self.registry.initial_population()
            if item.seed_id == seed_id
        )
        candidates = self.output_dir / "candidates"
        spec = write_candidate(seed, candidates / f"{seed_id}.json")
        task_name = f"mogpu_{seed_id}_{spec.ast_hash[:12]}"
        trial_dir = self.output_dir / "search" / task_name
        train_command = [
            "python",
            "src/train.py",
            "--config-name=unlearn.yaml",
            f"experiment={self.recipe['experiment']}",
            "trainer=MOGPU",
            f"trainer.method_args.candidate_spec_path={candidates / f'{seed_id}.json'}",
            f"task_name={task_name}",
            f"paths.output_dir={trial_dir}",
        ]
        subprocess.run(train_command, check=True, cwd=self.recipe["repo_root"])
        eval_dir = trial_dir / "evals"
        eval_command = [
            "python",
            "src/eval.py",
            f"experiment={self.recipe['eval_experiment']}",
            f"model.model_args.pretrained_model_name_or_path={trial_dir}",
            f"paths.output_dir={eval_dir}",
        ]
        subprocess.run(eval_command, check=True, cwd=self.recipe["repo_root"])
        summary = eval_dir / "TOFU_SUMMARY.json"
        metrics = json.loads(summary.read_text(encoding="utf-8"))
        fq = metrics["forget_quality"]
        entry = {
            "candidate_hash": spec.ast_hash,
            "parent_hashes": parents or [],
            "generation": generation,
            "operator": "seed",
            "config_hash": self.config_hash,
            "model": self.recipe["model"],
            "tofu_split": self.recipe["tofu_split"],
            "training_seed": self.recipe["training_seed"],
            "lora_rank": None,
            "budget": self.recipe["budget"],
            "static_gate": "passed",
            "output_dir": str(trial_dir),
            "evaluator_json_path": str(summary),
            "forget_quality": fq,
            "fq_feasible": fq >= 0.05,
            "model_utility": metrics.get("model_utility"),
            "tier": "search",
            "seed": self.registry.provenance_snapshot(seed_id),
            "explanation_card": render_card(spec, seed.data["source_method"]),
        }
        Ledger(self.output_dir, "search").append(entry)
        return entry
