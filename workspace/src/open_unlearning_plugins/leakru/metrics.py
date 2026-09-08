"""LeakRU metrics registered through the official UnlearningMetric decorator."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals
from open_unlearning_plugins.leakru.attacks import (
    focus_on_key_generate,
    greedy_generate,
    prepare_int4_model,
    probab_generate,
    quantization_generate,
)
from open_unlearning_plugins.leakru.scoring import (
    LOGIC_TYPES,
    answer_matches,
    any_trial_matches,
    forget_quality,
    fq_by_group,
    mean_correct,
    recovery_rate,
    select_s_suc,
)

logger = logging.getLogger("evaluator")


def _row_lookup(data):
    lookup = {}
    for i in range(len(data.data)):
        row = data.data[i]
        lookup[int(row["index"])] = row
        lookup[str(int(row["index"]))] = row
    return lookup


def _summarize(value_by_index):
    rows = list(value_by_index.values())
    by_type = defaultdict(list)
    by_split = defaultdict(list)
    for rec in rows:
        by_type[rec.get("logic_type")].append(rec)
        by_split[rec.get("split")].append(rec)
    return {
        "agg_value": mean_correct(rows),
        "by_logic_type": {k: mean_correct(v) for k, v in by_type.items() if k},
        "by_split": {k: mean_correct(v) for k, v in by_split.items() if k},
        "value_by_index": value_by_index,
    }


def _eval_generate_batch(model, tokenizer, batch, generation_args):
    texts = greedy_generate(model, tokenizer, batch, generation_args)
    return [{"generation": t} for t in texts]


@unlearning_metric(name="leakru_acc")
def leakru_acc(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    tokenizer = kwargs["tokenizer"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        _eval_generate_batch,
        {"tokenizer": tokenizer, "generation_args": generation_args},
        "LeakRU accuracy",
    )
    lookup = _row_lookup(data)
    value_by_index = {}
    for idx, rec in scores_by_index.items():
        row = lookup[int(idx)]
        generation = rec.get("generation", "")
        correct = answer_matches(generation, row["answer"], row.get("aliases"))
        value_by_index[str(int(idx))] = {
            "correct": bool(correct),
            "generation": generation,
            "gold": row["answer"],
            "aliases": list(row.get("aliases") or []),
            "logic_type": row["logic_type"],
            "split": row["split"],
            "question": row["question"],
        }
    return _summarize(value_by_index)


@unlearning_metric(name="leakru_fq")
def leakru_fq(model, **kwargs):
    unlearn = kwargs.get("pre_compute", {}).get("unlearn")
    if unlearn is None:
        logger.warning("leakru_fq missing pre_compute.unlearn (leakru_acc); agg_value=None")
        return {"agg_value": None}
    reference_logs = kwargs.get("reference_logs") or {}
    try:
        pretrained = reference_logs["pretrained_model_logs"]["pretrained"]
    except Exception:
        logger.warning(
            "pretrained_model_logs not provided in reference_logs, setting leakru_fq to None"
        )
        return {"agg_value": None}
    if pretrained is None:
        logger.warning("pretrained leakru_acc missing in reference logs, setting leakru_fq to None")
        return {"agg_value": None}

    by_type = fq_by_group(
        unlearn.get("by_logic_type") or {},
        pretrained.get("by_logic_type") or {},
    )
    for logic in LOGIC_TYPES:
        by_type.setdefault(logic, forget_quality(None, None))
    by_split = fq_by_group(
        unlearn.get("by_split") or {},
        pretrained.get("by_split") or {},
    )
    agg = forget_quality(unlearn.get("agg_value"), pretrained.get("agg_value"))
    if agg is None:
        logger.warning("leakru_fq undefined (pretrained Acc is 0 or missing)")
    return {
        "agg_value": agg,
        "by_logic_type": by_type,
        "by_split": by_split,
    }


def _score_generations(indices, generations, lookup, multi_trial=False):
    value_by_index = {}
    for idx, gen in zip(indices, generations):
        row = lookup[int(idx)]
        if multi_trial:
            correct = any_trial_matches(gen, row["answer"], row.get("aliases"))
            shown = gen
        else:
            correct = answer_matches(gen, row["answer"], row.get("aliases"))
            shown = gen
        value_by_index[str(int(idx))] = {
            "correct": bool(correct),
            "generation": shown,
            "gold": row["answer"],
            "aliases": list(row.get("aliases") or []),
            "logic_type": row["logic_type"],
            "split": row["split"],
        }
    return value_by_index


@unlearning_metric(name="leakru_rr")
def leakru_rr(model, **kwargs):
    reference_logs = kwargs.get("reference_logs") or {}
    try:
        pretrained = reference_logs["pretrained_model_logs"]["pretrained"]
    except Exception:
        logger.warning(
            "pretrained_model_logs not provided in reference_logs, setting leakru_rr to None"
        )
        return {"agg_value": None}
    if not pretrained or not pretrained.get("value_by_index"):
        logger.warning("pretrained leakru_acc.value_by_index missing, setting leakru_rr to None")
        return {"agg_value": None}

    s_suc = select_s_suc(pretrained["value_by_index"])
    if not s_suc:
        logger.warning("S_suc is empty (pretrained answered nothing correctly)")
        return {"agg_value": None, "s_suc_size": 0}

    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    tokenizer = kwargs["tokenizer"]
    generation_args = kwargs["generation_args"]
    template_args = kwargs.get("template_args")
    lookup = _row_lookup(data)
    suc_ints = {int(i) for i in s_suc}

    class _SucSubset:
        def __init__(self, inner, keep):
            self.inner = inner
            self.ids = [i for i in range(len(inner)) if int(inner.data[i]["index"]) in keep]

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return self.inner[self.ids[i]]

    subset = _SucSubset(data, suc_ints)
    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collator)

    def _indices_and_meta(mini_batch):
        idxs = mini_batch.pop("index").cpu().numpy().tolist()
        questions, answers, keys = [], [], []
        for idx in idxs:
            row = lookup[int(idx)]
            questions.append(row["question"])
            answers.append(row["answer"])
            keys.append(row.get("key_tokens"))
        return idxs, questions, answers, keys, mini_batch

    probab_map = {}
    focus_map = {}
    quant_map = {}
    for batch in dataloader:
        if "input_ids" in batch:
            batch = {"0": batch}
        mini = next(iter(batch.values()))
        idxs, questions, answers, keys, tensors = _indices_and_meta(mini)
        trials = probab_generate(model, tokenizer, tensors, generation_args)
        probab_map.update(_score_generations(idxs, trials, lookup, multi_trial=True))
        focus_texts = focus_on_key_generate(
            model,
            tokenizer,
            tensors,
            generation_args,
            questions=questions,
            key_tokens_list=keys,
            template_args=template_args,
            answers=answers,
        )
        focus_map.update(_score_generations(idxs, focus_texts, lookup))

    int4_model = prepare_int4_model(model)
    dataloader_q = DataLoader(subset, batch_size=batch_size, collate_fn=collator)
    for batch in dataloader_q:
        if "input_ids" in batch:
            batch = {"0": batch}
        mini = next(iter(batch.values()))
        idxs, _, _, _, tensors = _indices_and_meta(mini)
        texts = quantization_generate(int4_model, tokenizer, tensors, generation_args)
        quant_map.update(_score_generations(idxs, texts, lookup))

    attacks = {
        "probab": _summarize(probab_map),
        "focus_on_key": _summarize(focus_map),
        "quantization": _summarize(quant_map),
    }
    for name, payload in attacks.items():
        payload["agg_value"] = recovery_rate(payload["value_by_index"], s_suc)
        payload["s_suc_size"] = len(s_suc)

    rr_values = [attacks[k]["agg_value"] for k in attacks if attacks[k]["agg_value"] is not None]
    agg = float(np.mean(rr_values)) if rr_values else None
    return {"agg_value": agg, "s_suc_size": len(s_suc), "attacks": attacks}
