"""CPU tests for SAGE wiring, objective parsing, and NSGA offspring."""

import json
from pathlib import Path

from mogpu.operators import crossover, mutate_structure, mutate_thresholds, mutate_weights
from mogpu.search.archive import ParetoArchive
from mogpu.search import SearchController
from mogpu.search.objectives import apply_objectives, parse_summary
from mogpu.search.records import CandidateRecord
from mogpu.seed_registry import SeedRegistry
from mogpu.seeds import candidate_from_seed
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import probe_action_evidence, validate_candidate

ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return CandidateSpec.from_dict(
        {
            "atoms": ["EraseResidual", "RetainDrift"],
            "weights": [0.7, 0.3],
            "thresholds": {"kappa": 0.3, "tau": 0.05},
            "approved_variants": {"EraseResidual": "softplus", "RetainDrift": "huber"},
        }
    )


def test_parse_summary_skips_synthetic_optional_and_constraint(tmp_path):
    summary = tmp_path / "TOFU_SUMMARY.json"
    summary.write_text(
        json.dumps(
            {
                "forget_Truth_Ratio": 0.5,
                "model_utility": 0.58,
                "forget_quality": 0.02,
            }
        ),
        encoding="utf-8",
    )
    schema = {
        "forget_score": {"path": "forget_Truth_Ratio", "direction": "max"},
        "retain_utility": {"path": "model_utility", "direction": "max"},
        "negative_instability": {
            "path": "trajectory_stability",
            "direction": "max",
            "optional": True,
        },
        "negative_complexity": {
            "path": "mechanism_complexity",
            "direction": "min",
            "synthetic": True,
        },
        "forget_quality": {"path": "forget_quality", "direction": "max", "constraint": True},
    }
    parsed = parse_summary(summary, schema, 0.05)
    assert parsed["forget_score"] == 0.5
    assert parsed["raw_forget_quality"] == 0.02
    assert "negative_complexity" not in parsed
    record = CandidateRecord(
        candidate_hash="x",
        canonical_spec=_spec().to_dict(),
        generation=0,
    )
    apply_objectives(record, parsed, 0.05)
    assert record.fq_feasible is False
    assert record.objectives["negative_complexity"] == -2.0


def test_crossover_inherits_parent_weights():
    left = _spec()
    right = CandidateSpec.from_dict(
        {
            "atoms": ["EraseResidual", "RetainDrift", "SelectiveMargin"],
            "weights": [1.0, 1.0, 1.0],
            "thresholds": {"kappa": 0.3, "tau": 0.05, "temperature": 1.0},
            "approved_variants": {
                "EraseResidual": "softplus",
                "RetainDrift": "huber",
                "SelectiveMargin": "softplus_margin",
            },
        }
    )
    child = crossover(left, right)
    assert "SelectiveMargin" in [term.atom for term in child.terms]
    assert abs(sum(term.weight for term in child.terms) - 1.0) < 1e-8


def test_action_probe_matches_gate_contract():
    spec = validate_candidate(_spec())
    evidence = probe_action_evidence(spec)
    assert evidence["local_forget_action"] >= 0
    assert evidence["local_retain_action"] >= 0
    assert evidence["gradient_norm"] > 0


def test_dry_run_search_produces_summary(tmp_path):
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    recipe = {
        "repo_root": str(ROOT),
        "output_dir": str(tmp_path),
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "protocol_dir": str(ROOT / "configs/mogpu/search"),
        "retain_logs_path": str(tmp_path / "missing.json"),
        "fq_threshold": 0.05,
        "training_seed": 0,
        "lora_rank": None,
        "searchable": [],
        "dry_run": True,
        "budget": {"max_steps": 10},
    }
    controller = SearchController(registry, tmp_path, recipe)
    controller.protocol["population_size"] = 4
    controller.protocol["offspring_size"] = 4
    controller.protocol["max_generations"] = 2
    controller.protocol["archive_capacity"] = 4
    result = controller.run()
    assert "archive" in result
    assert (tmp_path / "search" / "ledger.jsonl").is_file()
    seed = next(
        item
        for item in registry.initial_population(benchmark="TOFU")
        if item.seed_id == "SIMNPO_ER_01"
    )
    mutated = mutate_weights(
        candidate_from_seed(seed), 0, controller.protocol["simplex"]
    )
    assert mutated.ast_hash
    assert controller.protocol["sage"]["f2"]["budget"]["max_steps"] == 10


def test_sage_overrides_replace_f2_budget(tmp_path):
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    recipe = {
        "repo_root": str(ROOT),
        "output_dir": str(tmp_path),
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "protocol_dir": str(ROOT / "configs/mogpu/search"),
        "retain_logs_path": str(tmp_path / "missing.json"),
        "fq_threshold": 0.05,
        "training_seed": 0,
        "lora_rank": None,
        "searchable": [],
        "dry_run": True,
        "budget": {"max_steps": 10},
        "sage_overrides": {
            "f2": {"budget": {"max_steps": 50}, "checkpoint_steps": [50]}
        },
    }
    controller = SearchController(registry, tmp_path, recipe)
    assert controller.protocol["sage"]["f2"]["budget"]["max_steps"] == 50
    assert controller.protocol["sage"]["f2"]["checkpoint_steps"] == [50]


def test_structure_and_threshold_mutations_stay_legal():
    spec = _spec()
    three = mutate_structure(spec, 1, ((1.0, 1.0), (1.0, 1.0, 1.0)))
    assert "SelectiveMargin" in [term.atom for term in three.terms]
    two = mutate_structure(three, 1, ((1.0, 1.0), (1.0, 1.0, 1.0)))
    assert [term.atom for term in two.terms] == ["EraseResidual", "RetainDrift"]
    shifted = mutate_thresholds(spec, 3, {"kappa": (0.3, 0.5), "tau": (0.02, 0.05)})
    assert dict(shifted.thresholds)["kappa"] in {0.3, 0.5}


def test_archive_keeps_infeasible_until_a_feasible_arrives():
    archive = ParetoArchive(2)
    weak = CandidateRecord(
        candidate_hash="weak",
        canonical_spec={},
        generation=0,
        status="passed",
        fq_feasible=False,
        constraint_violation=0.04,
        objectives={"retain_utility": 0.1},
    )
    strong = CandidateRecord(
        candidate_hash="strong",
        canonical_spec={},
        generation=0,
        status="passed",
        fq_feasible=False,
        constraint_violation=0.01,
        objectives={"retain_utility": 0.2},
    )
    archive.add(weak)
    archive.add(strong)
    assert {item["candidate_hash"] for item in archive.snapshot()} == {"weak", "strong"}
    feasible = CandidateRecord(
        candidate_hash="ok",
        canonical_spec={},
        generation=0,
        status="passed",
        fq_feasible=True,
        constraint_violation=0.0,
        objectives={"retain_utility": 0.05},
    )
    archive.add(feasible)
    assert [item["candidate_hash"] for item in archive.snapshot()] == ["ok"]
