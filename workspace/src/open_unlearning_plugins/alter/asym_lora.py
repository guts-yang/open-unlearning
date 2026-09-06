"""ALTER (AAAI 2026) reconstructed modules.

This is NOT official code. github.com/MastrOrigami/ALTER is an empty stub and
paper appendices A–G (IHL closed form, Tsallis q, expert count, LoRA sites)
are not in the public arXiv source. Defaults below are explicit reconstructions.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def tsallis_entropy(probs: torch.Tensor, q: float) -> torch.Tensor:
    """Token-wise Tsallis entropy. probs: [..., V]."""
    if abs(q - 1.0) < 1e-6:
        logp = torch.log(probs.clamp_min(1e-12))
        return -(probs * logp).sum(dim=-1)
    return (1.0 - (probs.clamp_min(1e-12).pow(q)).sum(dim=-1)) / (q - 1.0)


def ihl_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Assumed IHL: unlikelihood on gold + CE on next-best (gold masked).

    Paper: "suppress target, promote next-best"; closed form is Appendix B
    (unpublished). This formula is a documented reconstruction.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()

    valid = shift_labels.ne(ignore_index)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    gold_p = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    suppress = -torch.log((1.0 - gold_p).clamp_min(1e-12))

    masked_logits = shift_logits.clone()
    masked_logits.scatter_(-1, safe_labels.unsqueeze(-1), float("-inf"))
    next_best = masked_logits.argmax(dim=-1)
    promote = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        next_best.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)

    token_loss = suppress + promote
    denom = valid.float().sum().clamp_min(1.0)
    return (token_loss * valid.float()).sum() / denom


class AsymLoRALinear(nn.Module):
    """W = W0 + (Br + sum_d omega_f^d Bf^d) A x  with entropy-gated omega."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        n_forget_experts: int = 2,
        q: float = 1.5,
        high_entropy_cut: float = 1.2,
        tau_high: float = 0.8,
        tau_low: float = 0.01,
        top_k: int = 3,
        entropy_source: str | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.n_forget_experts = n_forget_experts
        self.q = q
        self.high_entropy_cut = high_entropy_cut
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.top_k = min(top_k, n_forget_experts)
        in_f = base.in_features
        out_f = base.out_features
        self.A = nn.Parameter(torch.empty(rank, in_f))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B_r = nn.Parameter(torch.zeros(out_f, rank))
        self.B_f = nn.Parameter(torch.zeros(n_forget_experts, out_f, rank))
        self.router = nn.Linear(1, n_forget_experts, bias=False)
        self._last_gate: torch.Tensor | None = None
        self._last_high_mask: torch.Tensor | None = None
        self._entropy_logits: torch.Tensor | None = None
        self.entropy_source = entropy_source

    def set_entropy_logits(self, logits: torch.Tensor | None) -> None:
        self._entropy_logits = logits

    def _token_entropy(self, x: torch.Tensor) -> torch.Tensor:
        if self._entropy_logits is not None:
            probs = F.softmax(self._entropy_logits, dim=-1)
            return tsallis_entropy(probs, self.q)
        # Fallback: feature-energy proxy when full vocab logits are unavailable.
        energy = x.pow(2).mean(dim=-1, keepdim=True)
        dummy = torch.stack([energy.squeeze(-1), 1.0 - energy.squeeze(-1).sigmoid()], dim=-1)
        dummy = dummy.softmax(dim=-1)
        return tsallis_entropy(dummy, self.q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base.weight, self.base.bias)
        entropy = self._token_entropy(x)
        high = entropy > self.high_entropy_cut
        tau = torch.where(
            high,
            torch.full_like(entropy, self.tau_high),
            torch.full_like(entropy, self.tau_low),
        )
        gate_logits = self.router(entropy.unsqueeze(-1)) / tau.unsqueeze(-1)
        gates = gate_logits.softmax(dim=-1)
        if not self.training and self.top_k < self.n_forget_experts:
            topv, topi = gates.topk(self.top_k, dim=-1)
            masked = torch.full_like(gates, 0.0)
            masked.scatter_(-1, topi, topv)
            gates = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        ax = F.linear(x, self.A)
        retain_delta = F.linear(ax, self.B_r)
        forget_delta = torch.einsum("...e,eoh,...h->...o", gates, self.B_f, ax)
        high_f = high.unsqueeze(-1).to(dtype=x.dtype)
        # Paper Eq. 9: high entropy uses forget experts; low entropy uses retain expert.
        delta = high_f * forget_delta + (1.0 - high_f) * retain_delta
        self._last_gate = gates
        self._last_high_mask = high
        return base_out + delta


def wrap_linear_modules(
    model: nn.Module,
    target_names: Iterable[str],
    **asym_kwargs,
) -> list[AsymLoRALinear]:
    wrapped = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) or t in name.split(".") for t in target_names):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model if parent_name == "" else model.get_submodule(parent_name)
        new_mod = AsymLoRALinear(module, **asym_kwargs)
        setattr(parent, child, new_mod)
        wrapped.append(new_mod)
    return wrapped


def iter_asym_modules(model: nn.Module) -> list[AsymLoRALinear]:
    return [m for m in model.modules() if isinstance(m, AsymLoRALinear)]
