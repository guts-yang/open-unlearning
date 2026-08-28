"""Token log-probability observables; NLL is used only via logp = -NLL."""

from __future__ import annotations

import torch
from torch.nn import functional


def model_inputs(inputs: dict) -> dict:
    return {key: inputs[key] for key in ("input_ids", "attention_mask", "labels")}


def mean_target_log_probs(model, inputs: dict) -> tuple[torch.Tensor, object]:
    """Return per-example mean log p(y|x), never raw logits as a DSL input."""
    inputs = model_inputs(inputs)
    outputs = model(**inputs)
    labels = inputs["labels"][..., 1:].contiguous()
    logits = outputs.logits[..., :-1, :].contiguous()
    log_probs = functional.log_softmax(logits, dim=-1)
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    counts = valid.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("MOGP-U requires at least one labeled target token per sample")
    return (selected * valid).sum(dim=-1) / counts, outputs


def deltas(model, reference_model, forget_inputs: dict, retain_inputs: dict):
    forget_logp, forget_outputs = mean_target_log_probs(model, forget_inputs)
    retain_logp, _ = mean_target_log_probs(model, retain_inputs)
    with torch.no_grad():
        ref_forget_logp, _ = mean_target_log_probs(reference_model, forget_inputs)
        ref_retain_logp, _ = mean_target_log_probs(reference_model, retain_inputs)
    return forget_logp - ref_forget_logp, retain_logp - ref_retain_logp, forget_outputs
