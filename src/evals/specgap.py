"""Core utilities for SpecGap post-hoc checkpoint auditing.

SpecGap measures the total-variation distance between the next-token
distributions of a frozen draft model and an unlearned target model.  This
module intentionally contains no Trainer integration: SpecGap is an audit
metric, not an unlearning objective.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.utils import IGNORE_INDEX, preprocess_chat_instance


CACHE_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE = 8192


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-serializable metadata."""
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def model_source_fingerprint(
    source: str, revision: str | None = None
) -> dict[str, Any]:
    """Fingerprint a local model directory or a remote model identifier.

    Local fingerprints include weight/config file size and modification time so
    that reusing a path for a different checkpoint cannot silently hit a stale
    draft cache.  Remote identifiers should be paired with a resolved commit
    revision whenever one is available.
    """
    path = Path(source).expanduser()
    if not path.is_dir():
        return {"source": source, "revision": revision or "main"}

    tracked_suffixes = {".json", ".model", ".safetensors", ".bin"}
    files = []
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix in tracked_suffixes:
            stat = file_path.stat()
            files.append(
                {
                    "name": file_path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {"source": str(path.resolve()), "revision": revision, "files": files}


def prepare_probe_item(
    tokenizer,
    template_args: Mapping[str, Any],
    question: str,
    answer: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize one QA pair and construct the shifted answer prediction mask.

    ``labels[j]`` is active exactly when token ``input_ids[j]`` belongs to the
    answer.  Since causal ``logits[j - 1]`` predicts that token, the mask for
    ``logits[:, :-1]`` is ``labels[1:] != IGNORE_INDEX``.
    """
    item = preprocess_chat_instance(
        tokenizer=tokenizer,
        template_config=dict(template_args),
        prompt_msgs=[question],
        response_msgs=[answer],
        max_length=max_length,
        predict_with_generate=False,
    )
    prediction_mask = item["labels"][1:].ne(IGNORE_INDEX)
    if not bool(prediction_mask.any()):
        raise ValueError("Probe sample has no answer tokens after tokenization")
    return {
        "input_ids": item["input_ids"],
        "attention_mask": item["attention_mask"],
        "prediction_mask": prediction_mask,
    }


@torch.no_grad()
def extract_answer_logits(
    model,
    item: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> torch.Tensor:
    """Run teacher forcing and return answer-position logits as ``[A, V]``."""
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(device)
    prediction_mask = item["prediction_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0, :-1]
    if logits.shape[0] != prediction_mask.shape[0]:
        raise ValueError(
            "Shifted answer mask does not align with model logits: "
            f"{prediction_mask.shape[0]} vs {logits.shape[0]}"
        )
    return logits[prediction_mask]


@torch.no_grad()
def min_distribution_overlap(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ``sum_v min(p(v), q(v))`` independently at each position."""
    if draft_logits.ndim != 2 or target_logits.ndim != 2:
        raise ValueError("SpecGap logits must both have shape [answer_tokens, vocab]")
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            f"Draft/target logits differ in shape: "
            f"{tuple(draft_logits.shape)} vs {tuple(target_logits.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    draft_logits = draft_logits.float()
    target_logits = target_logits.float()
    draft_lse = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
    target_lse = torch.logsumexp(target_logits, dim=-1, keepdim=True)
    overlap = torch.zeros(
        draft_logits.shape[0], dtype=torch.float32, device=draft_logits.device
    )

    vocab_size = draft_logits.shape[-1]
    for start in range(0, vocab_size, chunk_size):
        stop = min(start + chunk_size, vocab_size)
        draft_prob = (draft_logits[:, start:stop] - draft_lse).exp()
        target_prob = (target_logits[:, start:stop] - target_lse).exp()
        overlap.add_(torch.minimum(draft_prob, target_prob).sum(dim=-1))

    if not bool(torch.isfinite(overlap).all()):
        raise FloatingPointError("SpecGap overlap contains a non-finite value")
    tolerance = 2e-5
    if bool((overlap < -tolerance).any()) or bool((overlap > 1 + tolerance).any()):
        raise FloatingPointError(
            f"SpecGap overlap escaped [0, 1]: "
            f"min={overlap.min().item()}, max={overlap.max().item()}"
        )
    return overlap.clamp_(0.0, 1.0)


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile 95% CI for the sample mean."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("bootstrap values must be a non-empty one-dimensional array")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.size, size=(n_bootstrap, samples.size))
    means = samples[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def cohens_d(forget_values: Sequence[float], retain_values: Sequence[float]) -> float:
    """Compute pooled Cohen's d, positive when forget gaps are larger."""
    forget = np.asarray(forget_values, dtype=np.float64)
    retain = np.asarray(retain_values, dtype=np.float64)
    if forget.ndim != 1 or retain.ndim != 1:
        raise ValueError("Cohen's d inputs must be one-dimensional")
    if forget.size < 2 or retain.size < 2:
        raise ValueError("Cohen's d requires at least two values per group")

    numerator = (forget.size - 1) * forget.var(ddof=1)
    numerator += (retain.size - 1) * retain.var(ddof=1)
    pooled = np.sqrt(numerator / (forget.size + retain.size - 2))
    return float((forget.mean() - retain.mean()) / max(float(pooled), 1e-12))


def summarize_gaps(
    gaps: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize sample-level gaps for one probe split."""
    values = np.asarray(gaps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("SpecGap values must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("SpecGap values contain a non-finite value")
    low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95": [low, high],
    }


class DraftLogitsCache:
    """Versioned, per-sample disk cache for draft answer-position logits."""

    def __init__(self, root: str | Path, metadata: Mapping[str, Any]):
        full_metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            **dict(metadata),
        }
        self.metadata = full_metadata
        self.key = stable_hash(full_metadata)[:24]
        self.path = Path(root).expanduser() / self.key
        self.metadata_path = self.path / "metadata.json"

    def initialize(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            with self.metadata_path.open() as handle:
                existing = json.load(handle)
            if existing != self.metadata:
                raise ValueError(f"Draft cache metadata mismatch at {self.path}")
            return

        temporary = self.metadata_path.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.metadata_path)

    def shard_path(self, sample_index: int) -> Path:
        return self.path / f"{sample_index:06d}.pt"

    def contains(self, sample_index: int) -> bool:
        return self.shard_path(sample_index).is_file()

    def save(self, sample_index: int, logits: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(
                "Cached draft logits must have shape [answer_tokens, vocab]"
            )
        destination = self.shard_path(sample_index)
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(logits.detach().to(dtype=torch.bfloat16, device="cpu"), temporary)
        os.replace(temporary, destination)

    def load(self, sample_index: int) -> torch.Tensor:
        path = self.shard_path(sample_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing draft cache shard: {path}")
        logits = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError(f"Invalid draft cache shard: {path}")
        return logits
