from pathlib import Path

from mogpu.seed_registry import SeedRegistry
from mogpu.seeds import candidate_from_seed

ROOT = Path(__file__).resolve().parents[1]


def test_enabled_seeds_translate_to_closed_mechanism_candidates():
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    for record in registry.initial_population():
        candidate = candidate_from_seed(record)
        assert 2 <= len(candidate.terms) <= 3
        assert {term.atom for term in candidate.terms} <= {
            "EraseResidual",
            "RetainDrift",
            "SelectiveMargin",
        }


def test_simnpo_translation_is_smooth_erasure_with_retain_protection():
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    simnpo = next(
        record
        for record in registry.initial_population()
        if record.seed_id == "SIMNPO_ER_01"
    )
    candidate = candidate_from_seed(simnpo)
    assert {term.atom for term in candidate.terms} >= {"EraseResidual", "RetainDrift"}
    assert dict(candidate.thresholds)["kappa"] == 0.3
