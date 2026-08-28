"""Deduplicated Search-tier Pareto archive."""

from mogpu.search.nsga2 import environmental_selection
from mogpu.search.records import CandidateRecord


class ParetoArchive:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: dict[str, CandidateRecord] = {}

    def add(self, record: CandidateRecord) -> None:
        if record.tier != "search" or not record.fq_feasible:
            return
        self._items.setdefault(record.candidate_hash, record)
        chosen = environmental_selection(list(self._items.values()), self.capacity)
        self._items = {item.candidate_hash: item for item in chosen}

    def snapshot(self) -> list[dict]:
        return [
            item.to_dict()
            for item in sorted(
                self._items.values(), key=lambda item: item.candidate_hash
            )
        ]
