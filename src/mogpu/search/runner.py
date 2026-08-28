"""End-to-end Search / Validation / Sealed-test controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mogpu.search.controller import EvolutionController
from mogpu.search.evaluator import FitnessEvaluator
from mogpu.search.offspring import record_from_spec
from mogpu.search.records import CandidateRecord
from mogpu.search.sage_pareto import promote_f3
from mogpu.search.selection import select_knee
from mogpu.seed_registry import SeedRegistry
from mogpu.seeds import candidate_from_seed


def load_yaml(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_simplex(path: str | Path) -> tuple[tuple[float, ...], ...]:
    raw = load_yaml(path)
    return tuple(tuple(float(value) for value in row) for row in raw["weights"])


def load_threshold_grids(path: str | Path) -> dict[str, tuple[float, ...]]:
    raw = load_yaml(path)
    grids = raw.get("thresholds") or {}
    return {
        str(name): tuple(float(value) for value in values)
        for name, values in grids.items()
    }


def merge_sage(sage: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return sage
    merged = dict(sage)
    for stage, payload in overrides.items():
        if payload is None:
            continue
        base = dict(merged.get(stage) or {})
        overlay = dict(payload)
        budget = overlay.pop("budget", None) or {}
        cleaned_budget = {
            key: value for key, value in budget.items() if value is not None
        }
        if cleaned_budget:
            base["budget"] = {**(base.get("budget") or {}), **cleaned_budget}
        for key, value in overlay.items():
            if value is not None:
                base[key] = list(value) if isinstance(value, (list, tuple)) else value
        merged[stage] = base
    return merged


class SearchController:
    """Wire SAGE + NSGA-II onto frozen MOGPU training. Evolution is this class."""

    def __init__(
        self, registry: SeedRegistry, output_dir: str | Path, recipe: dict[str, Any]
    ):
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.recipe = recipe
        if not self.recipe.get("dry_run"):
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "Real MOGP-U search requires a CUDA-enabled PyTorch install"
                ) from error
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError(
                    "Real MOGP-U search requires at least two visible CUDA GPUs; "
                    "set dry_run=true explicitly for CPU-only orchestration tests"
                )
        self._validate_recipe()
        protocol_dir = Path(recipe["protocol_dir"])
        nsga = load_yaml(protocol_dir / "nsga2_sage_pareto.yaml")
        sage = merge_sage(
            load_yaml(protocol_dir / "sage_stages.yaml"),
            recipe.get("sage_overrides"),
        )
        frozen = load_yaml(protocol_dir / "frozen_search_protocol.yaml")
        simplex_path = protocol_dir / "weight_simplex.yaml"
        self.schema = load_yaml(protocol_dir / "objectives.yaml")
        self.fq_threshold = float(recipe.get("fq_threshold", frozen["fq_threshold"]))
        self.recipe.setdefault("cache_policy", frozen.get("cache_policy"))
        self.protocol = {
            **nsga,
            "sage": sage,
            "frozen": frozen,
            "simplex": load_simplex(simplex_path),
            "threshold_grids": load_threshold_grids(simplex_path),
            "training_seed": recipe.get("training_seed", 0),
        }
        self.protocol.update(
            {
                key: value
                for key, value in (recipe.get("nsga_overrides") or {}).items()
                if value is not None
            }
        )
        self.evaluator = FitnessEvaluator(recipe, self.schema, self.fq_threshold)
        self.evolution = EvolutionController(self.evaluator, self.protocol)

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
        logs = Path(self.recipe["retain_logs_path"])
        if not logs.is_file():
            logs = Path(self.recipe["repo_root"]) / self.recipe["retain_logs_path"]
        self.recipe["retain_logs_path"] = str(logs)
        if not self.recipe.get("dry_run") and not logs.is_file():
            raise FileNotFoundError(f"retain_logs_path missing: {logs}")

    def initial_records(self, benchmark: str = "TOFU") -> list[CandidateRecord]:
        records = []
        for seed in self.registry.initial_population(benchmark=benchmark):
            spec = candidate_from_seed(seed)
            record = record_from_spec(
                spec, 0, [], {"name": "seed", "seed_id": seed.seed_id}
            )
            records.append(record)
        return records

    def run(self) -> dict[str, Any]:
        parents = self.evolution.run(self.initial_records())
        promoted = self._run_f3()
        selected = None
        try:
            selected = select_knee(promoted or parents)
        except ValueError:
            selected = None
        return {
            "parents": [item.to_dict() for item in parents],
            "archive": self.evolution.archive.snapshot(),
            "f3": [item.to_dict() for item in promoted],
            "selected": None if selected is None else selected.to_dict(),
        }

    def run_seed(self, seed_id: str, generation: int = 0, parents: list[str] | None = None):
        seed = next(
            item
            for item in self.registry.initial_population()
            if item.seed_id == seed_id
        )
        spec = candidate_from_seed(seed)
        record = record_from_spec(
            spec, generation, parents or [], {"name": "seed", "seed_id": seed_id}
        )
        return self.evolution.run_generation([record])[0]

    def run_validation(
        self, record: CandidateRecord, num_train_epochs: int = 10, seeds: list[int] | None = None
    ) -> list[CandidateRecord]:
        seeds = seeds or [0, 1, 2]
        results = []
        original = dict(self.evaluator.recipe)
        self.evaluator.recipe["num_train_epochs"] = num_train_epochs
        self.evaluator.recipe["stage_budget"] = {"max_steps": -1}
        self.evaluator.recipe["checkpoint_steps"] = [10**9]
        for seed in seeds:
            clone = CandidateRecord(
                candidate_hash=record.candidate_hash,
                canonical_spec=record.canonical_spec,
                generation=record.generation,
                tier="validation",
                parent_hashes=record.parent_hashes,
                operator={"name": "validation", "seed": seed},
            )
            self.evaluator.recipe["training_seed"] = seed
            results.append(self.evaluator.evaluate(clone, max_steps=-1, seed=seed))
        self.evaluator.recipe = original
        return results

    def run_sealed_test(
        self, record: CandidateRecord, forget_split: str = "forget05"
    ) -> CandidateRecord:
        original = dict(self.evaluator.recipe)
        self.evaluator.recipe["forget_split"] = forget_split
        clone = CandidateRecord(
            candidate_hash=record.candidate_hash,
            canonical_spec=record.canonical_spec,
            generation=record.generation,
            tier="sealed_test",
            parent_hashes=record.parent_hashes,
            operator={"name": "sealed_test", "forget_split": forget_split},
        )
        result = self.evaluator.evaluate(clone, max_steps=-1, seed=0)
        self.evaluator.recipe = original
        return result

    def _run_f3(self) -> list[CandidateRecord]:
        if self.recipe.get("skip_f3"):
            return []
        stage = self.protocol["sage"]["f3"]
        capacity = int(stage["promotion_capacity"])
        seeds = list(stage["search_seeds"])
        max_steps = int(stage["budget"]["max_steps"])
        minimum = int(stage["minimum_sample_count"])
        archive = [
            CandidateRecord(
                candidate_hash=item["candidate_hash"],
                canonical_spec=item["canonical_spec"],
                generation=item["generation"],
                tier="search",
                fq_feasible=item["fq_feasible"],
                objectives=item["objectives"],
                status=item.get("status", "passed"),
            )
            for item in self.evolution.archive.snapshot()
        ][:capacity]
        promoted = []
        original = dict(self.evaluator.recipe)
        self.evaluator.recipe["stage_budget"] = dict(stage["budget"])
        self.evaluator.recipe["checkpoint_steps"] = [max_steps]
        for record in archive:
            samples = []
            for seed in seeds:
                clone = CandidateRecord(
                    candidate_hash=record.candidate_hash,
                    canonical_spec=record.canonical_spec,
                    generation=record.generation,
                    tier="search",
                    parent_hashes=record.parent_hashes,
                    operator={"name": "f3", "seed": seed},
                )
                self.evaluator.recipe["training_seed"] = seed
                samples.append(
                    self.evaluator.evaluate(clone, max_steps=max_steps, seed=seed)
                )
            if promote_f3(samples[-1], len(samples), minimum):
                promoted.append(samples[-1])
        self.evaluator.recipe = original
        return promoted
