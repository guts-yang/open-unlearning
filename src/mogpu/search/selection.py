"""Frozen Validation knee-point selection."""

import math

from mogpu.search.records import CandidateRecord


def select_knee(records: list[CandidateRecord]) -> CandidateRecord:
    eligible = [record for record in records if record.fq_feasible]
    if not eligible:
        raise ValueError(
            "No FQ-feasible candidate is eligible for Validation selection"
        )
    keys = sorted({key for record in eligible for key in record.objectives})
    low = {
        key: min(record.objectives.get(key, -math.inf) for record in eligible)
        for key in keys
    }
    high = {
        key: max(record.objectives.get(key, -math.inf) for record in eligible)
        for key in keys
    }

    def distance(record: CandidateRecord):
        return sum(
            (
                (high[key] - record.objectives.get(key, low[key]))
                / (high[key] - low[key])
            )
            ** 2
            for key in keys
            if high[key] > low[key]
        )

    return min(eligible, key=lambda record: (distance(record), record.candidate_hash))
