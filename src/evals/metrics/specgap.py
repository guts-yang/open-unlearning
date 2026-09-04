"""TOFU evaluator adapter for the SpecGap checkpoint audit."""

import gc
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.metrics.base import unlearning_metric
from evals.specgap import (
    DraftLogitsCache,
    cohens_d,
    extract_answer_logits,
    min_distribution_overlap,
    model_source_fingerprint,
    stable_hash,
    summarize_gaps,
)


logger = logging.getLogger("evaluator")


def _selected_positions(dataset, limit, seed):
    if limit is None or limit >= len(dataset):
        return list(range(len(dataset)))
    if limit <= 0:
        raise ValueError("SpecGap sample limit must be positive")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(dataset), size=limit, replace=False).tolist())


def _probe_item(dataset, position):
    item = dataset[position]
    prediction_mask = item["labels"][1:].ne(-100)
    if not bool(prediction_mask.any()):
        raise ValueError(f"SpecGap sample {position} has no answer tokens")
    return {
        "input_ids": item["input_ids"],
        "attention_mask": item["attention_mask"],
        "prediction_mask": prediction_mask,
        "dataset_index": int(item.get("index", position)),
    }


def _cache_metadata(
    dataset,
    positions,
    split,
    draft_model_path,
    draft_model,
    tokenizer,
    template_args,
):
    dataset_fingerprint = getattr(getattr(dataset, "data", None), "_fingerprint", None)
    return {
        "metric": "SpecGap-evaluator",
        "draft": model_source_fingerprint(
            draft_model_path, getattr(draft_model.config, "_commit_hash", None)
        ),
        "tokenizer_hash": stable_hash(
            {
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
            }
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "split": split,
        "positions": positions,
        "max_length": getattr(dataset, "max_length", None),
        "template": template_args,
    }


def _validate_tokenizer(tokenizer, draft_model_path):
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Draft and target tokenizer vocabularies differ")
    if tokenizer.special_tokens_map != draft_tokenizer.special_tokens_map:
        raise ValueError("Draft and target tokenizer special tokens differ")


@unlearning_metric(name="specgap")
def specgap(model, **kwargs):
    data = kwargs["data"]
    if not isinstance(data, dict) or set(data) != {"forget", "retain"}:
        raise ValueError("SpecGap metric requires forget and retain datasets")

    tokenizer = kwargs["tokenizer"]
    template_args = kwargs.get("template_args") or {}
    draft_model_path = kwargs.get("draft_model_path")
    if not draft_model_path:
        raise ValueError("SpecGap metric requires draft_model_path")
    seed = int(kwargs.get("seed", 0))
    retain_n = kwargs.get("retain_n", 400)
    chunk_size = int(kwargs.get("chunk_size", 8192))
    n_bootstrap = int(kwargs.get("n_bootstrap", 1000))
    cache_dir = kwargs.get(
        "cache_dir", "/root/autodl-tmp/saves/specgap/eval_cache"
    )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    _validate_tokenizer(tokenizer, draft_model_path)
    selections = {
        "forget": _selected_positions(data["forget"], None, seed),
        "retain": _selected_positions(data["retain"], retain_n, seed),
    }

    logger.info("Loading frozen SpecGap draft model from %s", draft_model_path)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        torch_dtype=model.dtype,
        device_map={"": model.device},
    )
    draft_model.eval()
    if draft_model.config.vocab_size != model.config.vocab_size:
        raise ValueError(
            "Draft/target model vocab mismatch: "
            f"{draft_model.config.vocab_size} vs {model.config.vocab_size}"
        )

    caches = {}
    for split, dataset in data.items():
        cache = DraftLogitsCache(
            cache_dir,
            _cache_metadata(
                dataset,
                selections[split],
                split,
                draft_model_path,
                draft_model,
                tokenizer,
                template_args,
            ),
        )
        cache.initialize()
        caches[split] = cache
        for ordinal, position in enumerate(selections[split]):
            if cache.contains(ordinal):
                continue
            item = _probe_item(dataset, position)
            logits = extract_answer_logits(draft_model, item, model.device)
            cache.save(ordinal, logits)

    del draft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    split_results = {}
    value_by_index = {}
    for split, dataset in data.items():
        samples = {}
        for ordinal, position in enumerate(selections[split]):
            item = _probe_item(dataset, position)
            target_logits = extract_answer_logits(model, item, model.device)
            draft_logits = caches[split].load(ordinal).to(model.device)
            overlap = min_distribution_overlap(
                draft_logits, target_logits, chunk_size=chunk_size
            )
            gap = float(1.0 - overlap.mean().item())
            samples[str(item["dataset_index"])] = {
                "specgap": gap,
                "answer_tokens": int(overlap.numel()),
            }
            del target_logits, draft_logits, overlap

        gaps = [sample["specgap"] for sample in samples.values()]
        split_results[split] = summarize_gaps(
            gaps, n_bootstrap=n_bootstrap, seed=seed
        )
        value_by_index[split] = samples

    forget_gaps = [
        sample["specgap"] for sample in value_by_index["forget"].values()
    ]
    retain_gaps = [
        sample["specgap"] for sample in value_by_index["retain"].values()
    ]
    aggregate = {
        "forget": split_results["forget"],
        "retain": split_results["retain"],
        "mean_difference": (
            split_results["forget"]["mean"] - split_results["retain"]["mean"]
        ),
        "cohens_d": cohens_d(forget_gaps, retain_gaps),
    }
    return {
        "agg_value": aggregate,
        "value_by_index": value_by_index,
        "settings": {
            "draft_model_path": draft_model_path,
            "retain_n": retain_n,
            "seed": seed,
            "chunk_size": chunk_size,
            "n_bootstrap": n_bootstrap,
            "cache_keys": {split: cache.key for split, cache in caches.items()},
        },
    }
