"""Pure constrained NSGA-II primitives; all objectives are maximized."""

from __future__ import annotations

import math
import random

from mogpu.search.records import CandidateRecord


def constrained_dominates(
    left: CandidateRecord, right: CandidateRecord, tolerance: float = 1e-12
) -> bool:
    if left.fq_feasible != right.fq_feasible:
        return left.fq_feasible
    if not left.fq_feasible:
        return left.constraint_violation < right.constraint_violation - tolerance
    keys = sorted(set(left.objectives) | set(right.objectives))
    no_worse = all(
        left.objectives.get(key, -math.inf)
        >= right.objectives.get(key, -math.inf) - tolerance
        for key in keys
    )
    strictly_better = any(
        left.objectives.get(key, -math.inf)
        > right.objectives.get(key, -math.inf) + tolerance
        for key in keys
    )
    return no_worse and strictly_better


def fast_non_dominated_sort(
    population: list[CandidateRecord],
) -> list[list[CandidateRecord]]:
    dominated = [[] for _ in population]
    counts = [0] * len(population)
    fronts: list[list[int]] = [[]]
    for index, item in enumerate(population):
        for other_index, other in enumerate(population):
            if constrained_dominates(item, other):
                dominated[index].append(other_index)
            elif constrained_dominates(other, item):
                counts[index] += 1
        if counts[index] == 0:
            item.nsga_rank = 0
            fronts[0].append(index)
    level = 0
    while level < len(fronts) and fronts[level]:
        following = []
        for index in fronts[level]:
            for other_index in dominated[index]:
                counts[other_index] -= 1
                if counts[other_index] == 0:
                    population[other_index].nsga_rank = level + 1
                    following.append(other_index)
        level += 1
        fronts.append(following)
    return [[population[index] for index in front] for front in fronts if front]


def crowding_distance(front: list[CandidateRecord]) -> None:
    if not front:
        return
    for item in front:
        item.crowding_distance = 0.0
    keys = sorted({key for item in front for key in item.objectives})
    for key in keys:
        ordered = sorted(
            front,
            key=lambda item: (item.objectives.get(key, -math.inf), item.candidate_hash),
        )
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        low, high = (
            ordered[0].objectives.get(key, 0.0),
            ordered[-1].objectives.get(key, 0.0),
        )
        if not math.isfinite(low) or not math.isfinite(high) or high == low:
            continue
        for index in range(1, len(ordered) - 1):
            ordered[index].crowding_distance += (
                ordered[index + 1].objectives.get(key, low)
                - ordered[index - 1].objectives.get(key, low)
            ) / (high - low)


def unique_records(population: list[CandidateRecord]) -> list[CandidateRecord]:
    unique: list[CandidateRecord] = []
    seen: set[str] = set()
    for item in population:
        if item.candidate_hash in seen:
            continue
        seen.add(item.candidate_hash)
        unique.append(item)
    return unique


def environmental_selection(
    population: list[CandidateRecord], size: int
) -> list[CandidateRecord]:
    selected = []
    for front in fast_non_dominated_sort(unique_records(population)):
        crowding_distance(front)
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            selected.extend(
                sorted(
                    front,
                    key=lambda item: (-item.crowding_distance, item.candidate_hash),
                )[: size - len(selected)]
            )
            break
    return selected


def binary_tournament(
    population: list[CandidateRecord], rng: random.Random
) -> CandidateRecord:
    first, second = rng.sample(population, 2)
    return min(
        (first, second),
        key=lambda item: (
            item.nsga_rank or 0,
            -item.crowding_distance,
            item.candidate_hash,
        ),
    )
