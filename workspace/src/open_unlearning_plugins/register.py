from __future__ import annotations

from typing import Iterable, Sequence, Type


PLUGIN_TRAINERS: list[Type] = []


def _load_plugin_trainers() -> list[Type]:
    from open_unlearning_plugins.alter.trainer import ALTER
    from open_unlearning_plugins.baldro.npo import DrNPO, GroupNPO
    from open_unlearning_plugins.baldro.satimp import DrSatImp, GroupSatImp
    from open_unlearning_plugins.baldro.simnpo import DrSimNPO, GroupSimNPO
    from open_unlearning_plugins.blade.lora_bial import LoRABiAL
    from open_unlearning_plugins.blade.lora_bial_adaptive import LoRABiALAdaptive
    from open_unlearning_plugins.grom.trainer import GROM

    return [
        LoRABiAL,
        LoRABiALAdaptive,
        GROM,
        DrNPO,
        GroupNPO,
        DrSimNPO,
        GroupSimNPO,
        DrSatImp,
        GroupSatImp,
        ALTER,
    ]


def register_workspace_trainers(names: Sequence[str] | None = None) -> list[str]:
    import trainer as trainer_mod

    trainers = _load_plugin_trainers()
    PLUGIN_TRAINERS.clear()
    PLUGIN_TRAINERS.extend(trainers)
    allow = set(names) if names is not None else None
    registered = []
    for cls in trainers:
        if allow is not None and cls.__name__ not in allow:
            continue
        trainer_mod._register_trainer(cls)
        registered.append(cls.__name__)
    return registered


def registered_handler_names() -> Iterable[str]:
    return [cls.__name__ for cls in _load_plugin_trainers()]


def register_workspace_evaluators() -> list[str]:
    """Register LeakRU evaluator, metrics, and dataset on official registries."""
    import data as data_mod
    import evals as evals_mod
    import evals.metrics as metrics_mod
    from open_unlearning_plugins.leakru.data import LeakRUDataset
    from open_unlearning_plugins.leakru.evaluator import LeakRUEvaluator
    from open_unlearning_plugins.leakru.metrics import leakru_acc, leakru_fq, leakru_rr

    evals_mod._register_evaluator(LeakRUEvaluator)
    data_mod._register_data(LeakRUDataset)
    for metric in (leakru_acc, leakru_fq, leakru_rr):
        metrics_mod._register_metric(metric)
    return [
        "LeakRUEvaluator",
        "LeakRUDataset",
        "leakru_acc",
        "leakru_fq",
        "leakru_rr",
    ]
