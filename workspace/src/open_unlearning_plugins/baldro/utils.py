"""BalDRO grouping helpers ported from nxZhai/BalDRO@5a93ede."""

from __future__ import annotations

import math

import torch

from trainer.utils import compute_batch_nll


def dv_aggregate(per_sample_loss, beta_dv):
    return beta_dv * (
        torch.logsumexp(per_sample_loss / beta_dv, dim=0)
        - math.log(per_sample_loss.numel())
    )


def compute_forget_group(model, forget_inputs, sampling_ratio=0.5):
    per_seq_nll, _ = compute_batch_nll(model, forget_inputs)
    valid_token_counts = (forget_inputs["labels"][..., 1:] != -100).sum(dim=-1)
    avg_nll = per_seq_nll / valid_token_counts.clamp(min=1)

    hardness = -avg_nll
    batch_size = hardness.size(0)
    k = max(1, int(batch_size * sampling_ratio))
    top_idx = torch.topk(hardness, k=k, largest=True, sorted=False).indices

    selected_forget_inputs = {
        "input_ids": forget_inputs["input_ids"].index_select(0, top_idx),
        "attention_mask": forget_inputs["attention_mask"].index_select(0, top_idx),
        "labels": forget_inputs["labels"].index_select(0, top_idx),
    }
    return selected_forget_inputs, k, top_idx


def compute_retain_group(retain_inputs, k):
    retain_b = retain_inputs["input_ids"].size(0)
    k_retain = min(k, retain_b)
    rand_idx = torch.randperm(retain_b, device=retain_inputs["input_ids"].device)[
        :k_retain
    ]
    return {
        "input_ids": retain_inputs["input_ids"].index_select(0, rand_idx),
        "attention_mask": retain_inputs["attention_mask"].index_select(0, rand_idx),
        "labels": retain_inputs["labels"].index_select(0, rand_idx),
    }
