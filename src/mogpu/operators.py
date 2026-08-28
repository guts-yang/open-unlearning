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


def mutate_thresholds(
    spec: CandidateSpec, seed: int, grids: dict[str, tuple[float, ...]]
) -> CandidateSpec:
    rng = random.Random(seed)
    data = spec.to_dict()
    keys = [name for name, values in grids.items() if values]
    if not keys:
        return spec
    name = rng.choice(keys)
    current = data.setdefault("thresholds", {}).get(name)
    options = [value for value in grids[name] if value != current] or list(grids[name])
    data["thresholds"][name] = float(rng.choice(options))
    return validate_candidate(CandidateSpec.from_dict(data))


def mutate_structure(
    spec: CandidateSpec, seed: int, simplex: tuple[tuple[float, ...], ...]
) -> CandidateSpec:
    rng = random.Random(seed)
    data = spec.to_dict()
    atoms = list(data["atoms"])
    variants = dict(data.get("approved_variants") or {})
    if "SelectiveMargin" in atoms:
        index = atoms.index("SelectiveMargin")
        atoms.pop(index)
        variants.pop("SelectiveMargin", None)
    else:
        atoms.append("SelectiveMargin")
        variants["SelectiveMargin"] = "softplus_margin"
        data.setdefault("thresholds", {}).setdefault("temperature", 1.0)
    options = [weights for weights in simplex if len(weights) == len(atoms)]
    if not options:
        raise ValueError("No approved discrete simplex for this candidate structure")
    data["atoms"] = atoms
    data["weights"] = list(rng.choice(options))
    data["approved_variants"] = variants
    return validate_candidate(CandidateSpec.from_dict(data))


def mutate_candidate(spec: CandidateSpec, seed: int, protocol: dict) -> CandidateSpec:
    rng = random.Random(seed)
    simplex = protocol["simplex"]
    grids = protocol.get("threshold_grids") or {}
    spec = mutate_weights(spec, rng.randint(0, 2**31 - 1), simplex)
    if grids and rng.random() < 0.5:
        spec = mutate_thresholds(spec, rng.randint(0, 2**31 - 1), grids)
    if rng.random() < float(protocol.get("structure_mutation_probability", 0.25)):
        spec = mutate_structure(spec, rng.randint(0, 2**31 - 1), simplex)
    return spec


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
