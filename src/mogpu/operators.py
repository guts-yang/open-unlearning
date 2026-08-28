"""Closed genetic operators over mechanism terms, never training recipes."""

from __future__ import annotations

import random

from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def mutate_weights(spec: CandidateSpec, seed: int) -> CandidateSpec:
    rng = random.Random(seed)
    data = spec.to_dict()
    data["weights"] = [
        max(0.0, weight + rng.uniform(-0.15, 0.15)) for weight in data["weights"]
    ]
    return validate_candidate(CandidateSpec.from_dict(data))


def crossover(left: CandidateSpec, right: CandidateSpec) -> CandidateSpec:
    atoms = list(dict.fromkeys(left.to_dict()["atoms"] + right.to_dict()["atoms"]))[:3]
    if len(atoms) < 2:
        raise ValueError("Crossover cannot produce fewer than two mechanism terms")
    source = left.to_dict()
    source["atoms"] = atoms
    source["weights"] = [1.0] * len(atoms)
    source["approved_variants"] = {
        atom: dict(left.approved_variants + right.approved_variants).get(
            atom, "softplus"
        )
        for atom in atoms
    }
    return validate_candidate(CandidateSpec.from_dict(source))
