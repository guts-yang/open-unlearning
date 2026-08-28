from pathlib import Path

import pytest
import yaml

from mogpu.seed_registry import SeedRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_default_catalog_only_exposes_enabled_seeds():
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    assert {record.seed_id for record in registry.initial_population()} == {
        "NPO_ER_01",
        "SIMNPO_ER_01",
        "LLMU_ER_01",
        "WGA_ER_01",
        "SATIMP_ER_01",
    }


@pytest.mark.parametrize(
    "field",
    [
        "source_repo_url",
        "source_repo_revision",
        "source_license",
        "source_code_files",
        "source_formula_locator",
    ],
)
def test_enabled_external_seed_requires_provenance(tmp_path, field):
    record = {
        "seed_id": "BAD",
        "source_method": "External",
        "source_paper_url": "https://example.com/paper",
        "source_repo_url": "https://github.com/example/repo",
        "source_repo_revision": "a" * 40,
        "source_license": "MIT",
        "source_code_files": ["loss.py"],
        "source_symbol_or_function": ["loss"],
        "source_formula_locator": "Eq. 1",
        "implementation_origin": "external_official_repo",
        "implementation_status": "enabled",
        "m_dsl_mapping": {"atoms": ["EraseResidual:softplus", "RetainDrift:huber"]},
    }
    record[field] = [] if field == "source_code_files" else ""
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump([record]), encoding="utf-8")
    with pytest.raises(ValueError):
        SeedRegistry.load(path)


def test_non_enabled_seed_never_enters_population(tmp_path):
    records = [
        {
            "seed_id": status,
            "source_method": "X",
            "implementation_origin": "openunlearning_builtin",
            "implementation_status": status,
            "m_dsl_mapping": {"atoms": []},
        }
        for status in ("pending_proof", "baseline_only", "excluded")
    ]
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(records), encoding="utf-8")
    assert SeedRegistry.load(path).initial_population() == []


def test_enabled_tpo_requires_complete_fixed_mask_proof(tmp_path):
    record = {
        "seed_id": "TPO",
        "source_method": "TPO",
        "source_paper_url": "https://example.com/paper",
        "source_repo_url": "https://github.com/example/repo",
        "source_repo_revision": "a" * 40,
        "source_license": "MIT",
        "source_code_files": ["loss.py"],
        "source_symbol_or_function": ["loss"],
        "source_formula_locator": "Eq. 1",
        "implementation_origin": "external_official_repo",
        "implementation_status": "enabled",
        "m_dsl_mapping": {"atoms": ["EraseResidual:softplus", "RetainDrift:huber"]},
        "proof_status": "pending_gradient_contract",
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="mask provenance"):
        SeedRegistry.load(path)
