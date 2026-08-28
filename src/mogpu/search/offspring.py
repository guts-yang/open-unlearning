"""Mechanism-term offspring; training recipes are never genes."""

from __future__ import annotations

import random

from mogpu.operators import crossover, mutate_weights
from mogpu.search.nsga2 import binary_tournament
from mogpu.search.records import CandidateRecord
from mogpu.search.sage_pareto import spec_from_record


def record_from_spec(
    spec, generation: int, parent_hashes: list[str], operator: dict
) -> CandidateRecord:
    data = spec.to_dict()
    return CandidateRecord(
        candidate_hash=spec.ast_hash,
        canonical_spec=data,
        generation=generation,
        parent_hashes=parent_hashes,
        operator=operator,
    )


def make_offspring(
    parents: list[CandidateRecord],
    generation: int,
    protocol: dict,
    rng: random.Random,
) -> list[CandidateRecord]:
    if len(parents) < 2:
        raise ValueError("Need at least two parents for NSGA-II offspring")
    simplex = protocol["simplex"]
    children: list[CandidateRecord] = []
    attempts = 0
    target = protocol["offspring_size"]
    while len(children) < target and attempts < target * 8:
        attempts += 1
        first = binary_tournament(parents, rng)
        second = binary_tournament(parents, rng)
        left = spec_from_record(first)
        right = spec_from_record(second)
        operator = {"name": "clone"}
        hashes = [first.candidate_hash]
        spec = left
        if rng.random() < protocol["crossover_probability"]:
            try:
                spec = crossover(left, right)
                operator = {"name": "crossover"}
                hashes = [first.candidate_hash, second.candidate_hash]
            except ValueError:
                spec = left
        if rng.random() < protocol["mutation_probability"]:
            spec = mutate_weights(spec, rng.randint(0, 2**31 - 1), simplex)
            operator = {**operator, "mutated": True}
        children.append(record_from_spec(spec, generation, hashes, operator))
    return children[:target]


def expand_initial(
    seeds: list[CandidateRecord], protocol: dict, rng: random.Random
) -> list[CandidateRecord]:
    population = list(seeds)
    seen = {item.candidate_hash for item in population}
    simplex = protocol["simplex"]
    guard = 0
    while len(population) < protocol["population_size"] and guard < 200:
        guard += 1
        parent = rng.choice(seeds)
        spec = mutate_weights(
            spec_from_record(parent), rng.randint(0, 2**31 - 1), simplex
        )
        if spec.ast_hash in seen:
            continue
        seen.add(spec.ast_hash)
        population.append(
            record_from_spec(
                spec,
                0,
                [parent.candidate_hash],
                {"name": "seed_mutate"},
            )
        )
    return population[: protocol["population_size"]]
