#!/usr/bin/env python3
"""Run the SpecGap post-hoc audit on TOFU checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from evals.specgap import (
    DEFAULT_CHUNK_SIZE,
    DraftLogitsCache,
    cohens_d,
    extract_answer_logits,
    min_distribution_overlap,
    model_source_fingerprint,
    prepare_probe_item,
    stable_hash,
    summarize_gaps,
)


DEFAULT_MODEL_CONFIG = "configs/model/Llama-3.2-1B-Instruct.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an unlearned checkpoint with full-vocabulary SpecGap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--draft", required=True, help="Pre-unlearning checkpoint")
    parser.add_argument("--target", required=True, help="Unlearned checkpoint")
    parser.add_argument(
        "--splits", nargs="+", default=["forget10", "retain90"], help="TOFU configs"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Samples per split; omitted means each split is evaluated in full",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling/bootstrap seed")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CONFIG,
        help="Model yaml supplying the project chat template",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "SPECGAP_CACHE_DIR", "/root/autodl-tmp/saves/specgap/cache"
        ),
    )
    parser.add_argument("--out", default="specprobe_result.json")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Probe draft against itself and require mean gap < 1e-3",
    )
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="Omit per-token overlap profiles from the output JSON",
    )
    return parser.parse_args()


def load_tokenizer(source: str):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def validate_target_tokenizer(draft_tokenizer, target: str) -> str:
    """Validate a target tokenizer when present, otherwise use the draft tokenizer."""
    try:
        target_tokenizer = load_tokenizer(target)
    except (OSError, ValueError):
        return "draft_fallback"

    if draft_tokenizer.get_vocab() != target_tokenizer.get_vocab():
        raise ValueError("Draft and target tokenizer vocabularies differ")
    if draft_tokenizer.special_tokens_map != target_tokenizer.special_tokens_map:
        raise ValueError("Draft and target tokenizer special tokens differ")
    return "validated"


def load_model(source: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    return model


def select_dataset(split: str, n: int | None, seed: int):
    dataset = load_dataset("locuslab/TOFU", name=split, split="train")
    if n is None or n >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())
    return dataset, indices


def cache_metadata(
    args: argparse.Namespace,
    split: str,
    dataset,
    indices: list[int],
    tokenizer,
    template_args: dict,
    revision: str | None,
) -> dict:
    return {
        "metric": "SpecGap",
        "draft": model_source_fingerprint(args.draft, revision),
        "tokenizer_hash": stable_hash(
            {
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
            }
        ),
        "dataset": "locuslab/TOFU",
        "dataset_fingerprint": dataset._fingerprint,
        "split": split,
        "indices": indices,
        "max_length": args.max_length,
        "template": template_args,
    }


def build_draft_caches(
    args: argparse.Namespace,
    model,
    tokenizer,
    template_args: dict,
    selections: dict,
) -> dict[str, DraftLogitsCache]:
    revision = getattr(model.config, "_commit_hash", None)
    caches = {}
    for split, selection in selections.items():
        dataset, indices = selection
        cache = DraftLogitsCache(
            args.cache_dir,
            cache_metadata(
                args,
                split,
                dataset,
                indices,
                tokenizer,
                template_args,
                revision,
            ),
        )
        cache.initialize()
        caches[split] = cache
        for ordinal, dataset_index in enumerate(indices):
            if cache.contains(ordinal):
                continue
            row = dataset[dataset_index]
            item = prepare_probe_item(
                tokenizer,
                template_args,
                row["question"],
                row["answer"],
                args.max_length,
            )
            logits = extract_answer_logits(model, item, args.device)
            cache.save(ordinal, logits)
    return caches


def audit_target(
    args: argparse.Namespace,
    model,
    tokenizer,
    template_args: dict,
    selections: dict,
    caches: dict[str, DraftLogitsCache],
) -> dict:
    split_results = {}
    for split, selection in selections.items():
        dataset, indices = selection
        sample_results = []
        for ordinal, dataset_index in enumerate(indices):
            row = dataset[dataset_index]
            item = prepare_probe_item(
                tokenizer,
                template_args,
                row["question"],
                row["answer"],
                args.max_length,
            )
            target_logits = extract_answer_logits(model, item, args.device)
            draft_logits = caches[split].load(ordinal).to(args.device)
            overlap = min_distribution_overlap(
                draft_logits, target_logits, chunk_size=args.chunk_size
            )
            profile = overlap.cpu().tolist()
            result = {
                "dataset_index": dataset_index,
                "answer_tokens": len(profile),
                "specgap": float(1.0 - overlap.mean().item()),
                "argmin_position": int(np.argmin(profile)),
            }
            if not args.no_profiles:
                result["overlap_profile"] = profile
            sample_results.append(result)

            del draft_logits, target_logits, overlap

        gaps = [sample["specgap"] for sample in sample_results]
        split_results[split] = {
            "summary": summarize_gaps(
                gaps, n_bootstrap=args.n_bootstrap, seed=args.seed
            ),
            "samples": sample_results,
            "cache_key": caches[split].key,
        }
    return split_results


def save_json(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.n is not None and args.n <= 0:
        raise ValueError("--n must be positive")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("SpecProbe requires a CUDA device for full-vocabulary audit")

    config = OmegaConf.load(args.model_config)
    template_args = OmegaConf.to_container(config.template_args, resolve=True)
    tokenizer = load_tokenizer(args.draft)
    tokenizer_status = validate_target_tokenizer(
        tokenizer, args.draft if args.self_test else args.target
    )
    selections = {
        split: select_dataset(split, args.n, args.seed) for split in args.splits
    }

    draft_model = load_model(args.draft, args.device)
    caches = build_draft_caches(args, draft_model, tokenizer, template_args, selections)
    del draft_model
    gc.collect()
    torch.cuda.empty_cache()

    target_source = args.draft if args.self_test else args.target
    target_model = load_model(target_source, args.device)
    tokenizer_size = len(tokenizer)
    if target_model.config.vocab_size != tokenizer_size:
        raise ValueError(
            f"Target model vocab size {target_model.config.vocab_size} differs from "
            f"tokenizer size {tokenizer_size}"
        )
    split_results = audit_target(
        args, target_model, tokenizer, template_args, selections, caches
    )

    comparison = None
    if len(args.splits) == 2:
        first, second = args.splits
        first_gaps = [sample["specgap"] for sample in split_results[first]["samples"]]
        second_gaps = [sample["specgap"] for sample in split_results[second]["samples"]]
        comparison = {
            "first_split": first,
            "second_split": second,
            "mean_difference": (
                split_results[first]["summary"]["mean"]
                - split_results[second]["summary"]["mean"]
            ),
            "cohens_d": cohens_d(first_gaps, second_gaps),
        }

    result = {
        "schema_version": 1,
        "metric": "SpecGap",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "draft": args.draft,
        "target": target_source,
        "tokenizer_status": tokenizer_status,
        "settings": {
            "splits": args.splits,
            "n": args.n,
            "seed": args.seed,
            "max_length": args.max_length,
            "chunk_size": args.chunk_size,
            "n_bootstrap": args.n_bootstrap,
            "model_config": args.model_config,
        },
        "splits": split_results,
        "comparison": comparison,
        "self_test": args.self_test,
    }
    if args.self_test:
        means = [value["summary"]["mean"] for value in split_results.values()]
        result["passed"] = max(means) < 1e-3
        save_json(args.out, result)
        if not result["passed"]:
            raise RuntimeError(f"SpecGap self-test failed with split means {means}")
    else:
        save_json(args.out, result)

    for split, value in split_results.items():
        summary = value["summary"]
        print(
            f"[{split}] SpecGap={summary['mean']:.6f} "
            f"CI95=[{summary['ci95'][0]:.6f}, {summary['ci95'][1]:.6f}] "
            f"n={summary['n']}"
        )
    if comparison is not None:
        print(
            f"[comparison] delta={comparison['mean_difference']:.6f} "
            f"Cohen's d={comparison['cohens_d']:.4f}"
        )
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
