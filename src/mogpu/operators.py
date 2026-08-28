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
    atoms = list(dict.fromkeys(left.to_dict()["atoms"] + right.to_dict()["atoms"]))[:3]
    if not {"EraseResidual", "RetainDrift"} <= set(atoms):
        raise ValueError("Crossover must retain EraseResidual and RetainDrift")
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
