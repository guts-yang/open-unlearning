"""Strict provenance registry for external MOGP-U mechanism seeds."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

STATUSES = {"enabled", "pending_proof", "baseline_only", "excluded"}
ORIGINS = {
    "openunlearning_builtin",
    "external_official_repo",
    "paper_only_pending_proof",
}
EXTERNAL_REQUIRED = (
    "source_paper_url",
    "source_repo_url",
    "source_repo_revision",
    "source_license",
    "source_code_files",
    "source_symbol_or_function",
    "source_formula_locator",
)


@dataclass(frozen=True)
class SeedRecord:
    data: dict[str, Any]

    @property
    def seed_id(self) -> str:
        return self.data["seed_id"]

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data)


class SeedRegistry:
    def __init__(self, records: list[SeedRecord], root: Path):
        self.records = records
        self.root = root

    @classmethod
    def load(cls, path: str | Path) -> SeedRegistry:
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or []
        records = raw.get("seeds", raw) if isinstance(raw, dict) else raw
        if not isinstance(records, list):
            raise TypeError("seed catalog must be a list or contain a seeds list")
        parsed = [SeedRecord(dict(record)) for record in records]
        registry = cls(parsed, Path(path).resolve().parents[2])
        registry._validate()
        return registry

    def _validate(self) -> None:
        ids = set()
        for record in self.records:
            data = record.data
            for key in (
                "seed_id",
                "source_method",
                "implementation_origin",
                "implementation_status",
                "m_dsl_mapping",
            ):
                if not data.get(key):
                    raise ValueError(
                        f"{record.seed_id if 'seed_id' in data else 'seed'} missing {key}"
                    )
            if data["seed_id"] in ids:
                raise ValueError(f"Duplicate seed_id: {data['seed_id']}")
            ids.add(data["seed_id"])
            if data["implementation_status"] not in STATUSES:
                raise ValueError(
                    f"Invalid implementation status: {data['implementation_status']}"
                )
            if data["implementation_origin"] not in ORIGINS:
                raise ValueError(
                    f"Invalid implementation origin: {data['implementation_origin']}"
                )
            if (
                data["implementation_origin"] == "paper_only_pending_proof"
                and data["implementation_status"] == "enabled"
            ):
                raise ValueError("paper-only seed cannot be enabled")
            if (
                data["implementation_origin"] == "external_official_repo"
                and data["implementation_status"] == "enabled"
            ):
                for key in EXTERNAL_REQUIRED:
                    if not data.get(key):
                        raise ValueError(
                            f"Enabled external seed {data['seed_id']} missing {key}"
                        )
                revision = str(data["source_repo_revision"]).lower()
                if revision in {"main", "master"}:
                    raise ValueError(
                        "source_repo_revision must be a commit SHA or release tag"
                    )
            atoms = data["m_dsl_mapping"].get("atoms", [])
            if data["implementation_status"] == "enabled" and (
                not 2 <= len(atoms) <= 3
                or any(
                    atom.split(":")[0]
                    not in {"EraseResidual", "RetainDrift", "SelectiveMargin"}
                    for atom in atoms
                )
            ):
                raise ValueError(
                    f"Enabled seed {data['seed_id']} has invalid M-DSL mapping"
                )
            if data["source_method"] in {"WGA", "SatImp"}:
                for key in (
                    "fixed_weight_manifest",
                    "fixed_weight_manifest_hash",
                    "weight_generation_rule",
                ):
                    if data["implementation_status"] == "enabled" and not data.get(key):
                        raise ValueError(f"{data['source_method']} seed missing {key}")
                if data["implementation_status"] == "enabled":
                    manifest = self.root / data["fixed_weight_manifest"]
                    if not manifest.is_file():
                        raise ValueError(
                            f"{data['source_method']} weight manifest does not exist"
                        )
                    actual_hash = sha256(manifest.read_bytes()).hexdigest()
                    if actual_hash != data["fixed_weight_manifest_hash"]:
                        raise ValueError(
                            f"{data['source_method']} weight manifest hash mismatch"
                        )
            if data["source_method"] in {"TPO", "LoKU"}:
                required = (
                    "target_token_mask_manifest",
                    "target_token_mask_manifest_hash",
                    "proof_status",
                )
                if data["implementation_status"] == "enabled" and not all(
                    data.get(key) for key in required
                ):
                    raise ValueError(
                        f"{data['source_method']} mask provenance is incomplete"
                    )

    def initial_population(
        self, mechanism: str | None = None, benchmark: str | None = None
    ):
        selected = [
            record
            for record in self.records
            if record.data["implementation_status"] == "enabled"
        ]
        if mechanism:
            selected = [
                record
                for record in selected
                if mechanism in str(record.data["m_dsl_mapping"])
            ]
        if benchmark:
            selected = [
                record
                for record in selected
                if benchmark in record.data.get("benchmark_scope", [])
            ]
        return selected

    def provenance_snapshot(self, seed_id: str) -> dict[str, Any]:
        for record in self.records:
            if record.seed_id == seed_id:
                return record.snapshot()
        raise KeyError(seed_id)
