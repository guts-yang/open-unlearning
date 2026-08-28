"""Single-candidate serial scheduling with protocol-scoped cache."""

from collections.abc import Callable

from mogpu.search.records import CandidateRecord


def copy_fitness(cached: CandidateRecord, record: CandidateRecord) -> CandidateRecord:
    record.fq_feasible = cached.fq_feasible
    record.constraint_violation = cached.constraint_violation
    record.objectives = dict(cached.objectives)
    record.stage = cached.stage
    record.status = cached.status
    record.payload = {**dict(cached.payload), **record.payload}
    return record


class SearchScheduler:
    def __init__(self, run_one: Callable, cache: dict | None = None):
        self.run_one = run_one
        self.cache = cache if cache is not None else {}

    def evaluate(self, key: tuple, record: CandidateRecord, *args, **kwargs):
        if key in self.cache:
            return copy_fitness(self.cache[key], record), True
        result = self.run_one(record, *args, **kwargs)
        self.cache[key] = result
        return result, False
