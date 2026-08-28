"""Immutable schema and canonical serialization for the MOGP-U DSL."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ALLOWED_ATOMS = {"EraseResidual", "RetainDrift", "SelectiveMargin"}
ALLOWED_VARIANTS = {
    "EraseResidual": {"softplus"},
    "RetainDrift": {"huber"},
    "SelectiveMargin": {"softplus_margin"},
}


@dataclass(frozen=True)
class MechanismTerm:
    atom: str
    weight: float
    variant: str


@dataclass(frozen=True)
class CandidateSpec:
    """A closed M-DSL candidate, never an arbitrary expression tree."""

    schema_version: int
    terms: tuple[MechanismTerm, ...]
    thresholds: tuple[tuple[str, float], ...]
    approved_variants: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateSpec:
        allowed = {
            "schema_version",
            "atoms",
            "weights",
            "thresholds",
            "approved_variants",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"CandidateSpec contains forbidden fields: {sorted(unknown)}"
            )
        atoms = value.get("atoms")
        weights = value.get("weights")
        variants = value.get("approved_variants", {})
        if not isinstance(atoms, list) or not isinstance(weights, list):
            raise TypeError("atoms and weights must be lists")
        if len(atoms) != len(weights):
            raise ValueError("atoms and weights must have the same length")
        terms = tuple(
            MechanismTerm(
                atom=str(atom),
                weight=float(weight),
                variant=str(variants.get(atom, _default_variant(str(atom)))),
            )
            for atom, weight in zip(atoms, weights)
        )
        thresholds = tuple(
            sorted(
                (str(key), float(number))
                for key, number in value.get("thresholds", {}).items()
            )
        )
        approved = tuple(
            sorted((str(key), str(name)) for key, name in variants.items())
        )
        return cls(
            int(value.get("schema_version", 1)), terms, thresholds, approved
        ).canonical()

    @classmethod
    def load(cls, path: str | Path) -> CandidateSpec:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def canonical(self) -> CandidateSpec:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported CandidateSpec schema version: {self.schema_version}"
            )
        if not 2 <= len(self.terms) <= 3:
            raise ValueError(
                "CandidateSpec must contain exactly 2 or 3 mechanism terms"
            )
        if len({term.atom for term in self.terms}) != len(self.terms):
            raise ValueError("A mechanism may appear at most once in a CandidateSpec")
        atom_names = {term.atom for term in self.terms}
        if not {"EraseResidual", "RetainDrift"} <= atom_names:
            raise ValueError("CandidateSpec must contain EraseResidual and RetainDrift")
        total = sum(term.weight for term in self.terms)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("CandidateSpec weights must have a finite positive sum")
        terms = []
        for term in self.terms:
            if term.atom not in ALLOWED_ATOMS:
                raise ValueError(f"Unsupported M-DSL atom: {term.atom}")
            if term.variant not in ALLOWED_VARIANTS[term.atom]:
                raise ValueError(f"Unsupported {term.atom} variant: {term.variant}")
            if not math.isfinite(term.weight) or term.weight < 0:
                raise ValueError(
                    "CandidateSpec weights must be finite and non-negative"
                )
            terms.append(MechanismTerm(term.atom, term.weight / total, term.variant))
        threshold_map = dict(self.thresholds)
        for key, number in threshold_map.items():
            if key not in {
                "kappa",
                "tau",
                "temperature",
                "epsilon",
            } or not math.isfinite(number):
                raise ValueError(f"Invalid threshold: {key}")
        return CandidateSpec(
            schema_version=1,
            terms=tuple(sorted(terms, key=lambda item: item.atom)),
            thresholds=tuple(sorted(threshold_map.items())),
            approved_variants=tuple(
                sorted((term.atom, term.variant) for term in terms)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atoms": [term.atom for term in self.terms],
            "weights": [float(f"{term.weight:.17g}") for term in self.terms],
            "thresholds": {key: value for key, value in self.thresholds},
            "approved_variants": {key: value for key, value in self.approved_variants},
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def ast_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def threshold(self, name: str, default: float) -> float:
        return dict(self.thresholds).get(name, default)


def _default_variant(atom: str) -> str:
    return next(iter(ALLOWED_VARIANTS.get(atom, {"invalid"})))
