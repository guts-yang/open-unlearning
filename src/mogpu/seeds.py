"""Translate audited seed mappings to frozen CandidateSpec files."""

from __future__ import annotations

import json
from pathlib import Path

from mogpu.seed_registry import SeedRecord
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def candidate_from_seed(seed: SeedRecord) -> CandidateSpec:
    mapping = seed.data["m_dsl_mapping"]
    atoms = [entry.split(":")[0] for entry in mapping["atoms"]]
    if len(atoms) < 2 or "EraseResidual" not in atoms:
        raise ValueError("Seed must form a legal E+R/S final candidate")
    parameters = seed.data.get("approved_parameters", {})
    thresholds = {
        name: values[0] if isinstance(values, list) else values
        for name, values in parameters.items()
        if name in {"kappa", "tau", "temperature", "epsilon"}
    }
    spec = CandidateSpec.from_dict(
        {
            "schema_version": 1,
            "atoms": atoms,
            "weights": [1.0] * len(atoms),
            "thresholds": thresholds,
            "approved_variants": {
                entry.split(":")[0]: entry.split(":")[1] if ":" in entry else "softplus"
                for entry in mapping["atoms"]
            },
        }
    )
    return validate_candidate(spec)


def write_candidate(seed: SeedRecord, destination: str | Path) -> CandidateSpec:
    spec = candidate_from_seed(seed)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return spec
