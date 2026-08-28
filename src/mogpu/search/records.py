"""Serializable records exchanged by the search layers."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    candidate_hash: str
    canonical_spec: dict[str, Any]
    generation: int
    tier: str = "search"
    stage: str = "F0"
    status: str = "pending"
    fq_feasible: bool = False
    constraint_violation: float = float("inf")
    objectives: dict[str, float] = field(default_factory=dict)
    parent_hashes: list[str] = field(default_factory=list)
    operator: dict[str, Any] = field(default_factory=dict)
    nsga_rank: int | None = None
    crowding_distance: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
