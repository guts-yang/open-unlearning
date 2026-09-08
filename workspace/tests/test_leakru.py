from __future__ import annotations

import json

from open_unlearning_plugins.leakru.data import load_leakru  # noqa: E402
from open_unlearning_plugins.leakru.scoring import (  # noqa: E402
    LOGIC_TYPES,
    PROBAB_TEMPERATURE,
    PROBAB_TOP_P,
    PROBAB_TRIALS,
    answer_matches,
    any_trial_matches,
    extract_key_tokens,
    focus_on_key_prompt,
    forget_quality,
    fq_by_group,
    recovery_rate,
    select_s_suc,
)
from open_unlearning_plugins.register import (  # noqa: E402
    register_workspace_evaluators,
    register_workspace_trainers,
)

import pytest


def _six_rows():
    rows = []
    for i, logic in enumerate(LOGIC_TYPES):
        rows.append(
            {
                "question": f"Q{logic} entity Name {i}?",
                "answer": f"Ans{logic}",
                "aliases": [f"Alias{logic}"],
                "logic_type": logic,
                "split": "multi-hop" if i % 2 else "single-hop",
            }
        )
    return rows


def test_evaluators_and_metrics_register_without_hf_download():
    pytest.importorskip("torch")
    from data import DATASET_REGISTRY
    from evals import EVALUATOR_REGISTRY
    from evals.metrics import METRICS_REGISTRY

    register_workspace_trainers()
    names = register_workspace_evaluators()
    assert "LeakRUEvaluator" in names
    assert "LeakRUEvaluator" in EVALUATOR_REGISTRY
    assert "LeakRUDataset" in DATASET_REGISTRY
    for metric in ("leakru_acc", "leakru_fq", "leakru_rr"):
        assert metric in METRICS_REGISTRY


def test_load_leakru_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="LeakRU data not found"):
        load_leakru(data_path=tmp_path / "does-not-exist.jsonl")


def test_load_leakru_missing_logic_type(tmp_path):
    path = tmp_path / "mquake.jsonl"
    row = _six_rows()[0]
    del row["logic_type"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_leakru(data_path=path)


def test_load_leakru_schema_and_per_type_limit(tmp_path):
    path = tmp_path / "mquake.jsonl"
    extra = _six_rows() + _six_rows()
    path.write_text(
        "\n".join(json.dumps(r) for r in extra) + "\n", encoding="utf-8"
    )
    rows = load_leakru(data_path=path, per_type_limit=1)
    assert len(rows) == 6
    assert {r["logic_type"] for r in rows} == set(LOGIC_TYPES)


def test_load_leakru_rejects_incomplete_types(tmp_path):
    path = tmp_path / "mquake.jsonl"
    path.write_text(json.dumps(_six_rows()[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing logic_type"):
        load_leakru(data_path=path)


def test_answer_matches_and_probab_any_trial():
    assert answer_matches("The capital is Paris.", "Paris", ["Lutetia"])
    assert answer_matches("aka Lutetia today", "Paris", ["Lutetia"])
    assert not answer_matches("I don't know", "Paris", ["Lutetia"])
    assert any_trial_matches(["nope", "Paris is the answer"], "Paris", [])
    assert not any_trial_matches(["nope", "maybe"], "Paris", [])


def test_forget_quality_formula():
    assert forget_quality(0.2, 0.5) == pytest.approx(0.6)
    assert forget_quality(0.5, 0.5) == pytest.approx(0.0)
    assert forget_quality(0.1, 0.0) is None
    grouped = fq_by_group({"HS": 0.2, "MT": 0.0}, {"HS": 0.4, "MT": 0.0})
    assert grouped["HS"] == pytest.approx(0.5)
    assert grouped["MT"] is None


def test_s_suc_and_recovery_rate():
    pretrained = {
        "0": {"correct": True},
        "1": {"correct": False},
        "2": {"correct": True},
    }
    s_suc = select_s_suc(pretrained)
    assert s_suc == ["0", "2"]
    attack = {
        "0": {"correct": True},
        "2": {"correct": False},
    }
    assert recovery_rate(attack, s_suc) == pytest.approx(0.5)
    assert recovery_rate(attack, []) is None


def test_focus_on_key_defaults():
    assert PROBAB_TEMPERATURE == 0.8
    assert PROBAB_TOP_P == 0.95
    assert PROBAB_TRIALS == 5
    tokens = extract_key_tokens("Who is Tom Cruise married to?", provided=None)
    assert "Tom Cruise" in tokens or "Cruise" in " ".join(tokens)
    prompt = focus_on_key_prompt("Q?", key_tokens=["France"], repeats=3)
    assert prompt.count("France") == 3


def test_hydra_compose_tofu_and_leakru(repo_root):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    ws = repo_root / "workspace" / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                f"hydra.searchpath=[file://{ws}]",
                "experiment=eval/tofu_leakru_mquake",
                "task_name=compose_test",
            ],
        )
    assert "tofu" in cfg.eval
    assert "leakru" in cfg.eval
    assert cfg.eval.leakru.handler == "LeakRUEvaluator"
    assert cfg.eval.tofu.handler == "TOFUEvaluator"
    assert cfg.eval.leakru.metrics.leakru_acc.handler == "leakru_acc"
    assert cfg.eval.leakru.metrics.leakru_fq.handler == "leakru_fq"
    assert cfg.eval.leakru.metrics.leakru_rr.handler == "leakru_rr"
    GlobalHydra.instance().clear()
