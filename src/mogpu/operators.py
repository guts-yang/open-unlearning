"""Closed genetic operators over mechanism terms, never training recipes."""

from __future__ import annotations

import random

from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def mutate_weights(
    spec: CandidateSpec, seed: int, simplex: tuple[tuple[float, ...], ...]
) -> CandidateSpec:
    rng = random.Random(seed)
    data = spec.to_dict()
    options = [weights for weights in simplex if len(weights) == len(data["atoms"])]
    if not options:
        raise ValueError("No approved discrete simplex for this candidate structure")
    data["weights"] = list(rng.choice(options))
    return validate_candidate(CandidateSpec.from_dict(data))


def crossover(left: CandidateSpec, right: CandidateSpec) -> CandidateSpec:
    left_data, right_data = left.to_dict(), right.to_dict()
    atoms = list(dict.fromkeys(left_data["atoms"] + right_data["atoms"]))[:3]
    if not {"EraseResidual", "RetainDrift"} <= set(atoms):
        raise ValueError("Crossover must retain EraseResidual and RetainDrift")
    inherited = {}
    for atom, weight in zip(left_data["atoms"], left_data["weights"]):
        inherited[atom] = weight
    for atom, weight in zip(right_data["atoms"], right_data["weights"]):
        inherited[atom] = (
            0.5 * (inherited[atom] + weight) if atom in inherited else weight
        )
    source = left_data
    source["atoms"] = atoms
    source["weights"] = [inherited[atom] for atom in atoms]
    source["thresholds"] = {
        **right_data.get("thresholds", {}),
        **left_data.get("thresholds", {}),
    }
    source["approved_variants"] = {
        atom: dict(left.approved_variants + right.approved_variants).get(
            atom, "softplus"
        )
        for atom in atoms
    }
    return validate_candidate(CandidateSpec.from_dict(source))
