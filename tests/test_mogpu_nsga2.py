import random

from mogpu.search.nsga2 import (
    binary_tournament,
    environmental_selection,
    fast_non_dominated_sort,
)
from mogpu.search.records import CandidateRecord


def record(name, fq, utility, violation=0.0):
    return CandidateRecord(
        candidate_hash=name,
        canonical_spec={},
        generation=0,
        fq_feasible=fq,
        constraint_violation=violation,
        objectives={"retain_utility": utility, "negative_complexity": -2.0},
    )


def test_feasible_beats_infeasible_regardless_of_utility():
    feasible = record("a", True, 0.1)
    infeasible = record("b", False, 0.99, 0.01)
    assert fast_non_dominated_sort([feasible, infeasible])[0] == [feasible]


def test_environmental_selection_deduplicates_hashes():
    population = [
        record("a", True, 0.2),
        record("a", True, 0.2),
        record("b", True, 0.4),
        record("b", True, 0.4),
    ]
    selected = environmental_selection(population, 4)
    assert {item.candidate_hash for item in selected} == {"a", "b"}
    assert len(selected) == 2


def test_environmental_selection_and_tournament_are_deterministic():
    population = [
        record("a", True, 0.2),
        record("b", True, 0.4),
        record("c", False, 1.0, 0.1),
    ]
    selected = environmental_selection(population, 2)
    assert {item.candidate_hash for item in selected} == {"a", "b"}
    assert (
        binary_tournament(selected, random.Random(7)).candidate_hash
        == binary_tournament(selected, random.Random(7)).candidate_hash
    )
