"""GROM unlearning targets. Pin Batorskq/GROM@6dd9591.

suppress: D[:, j] = -beta * alpha[g_j] * uhat[g_j]
rmu: rank-one corruption D = c * u * 1^T (formed as B without materialising D)
"""

from __future__ import annotations

import torch


def specificity_alpha(
    forget_counts: torch.Tensor, retain_counts: torch.Tensor
) -> torch.Tensor:
    ff = forget_counts.double()
    ff = ff / ff.sum()
    rf = retain_counts.double()
    rf = rf / rf.sum().clamp_min(1)
    return (1.0 - rf / (ff + 1e-12)).clamp(0, 1)


def unit_unembeddings(model) -> torch.Tensor:
    W = model.lm_head.weight.detach().float()
    return W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)


def suppression_D(
    uhat: torch.Tensor, alpha: torch.Tensor, gold: torch.Tensor, beta: float
) -> torch.Tensor:
    return -beta * alpha[gold].float().unsqueeze(0) * uhat[gold].T


def corruption_direction(d_out: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d_out, generator=g, dtype=torch.float64)
    return (u / u.norm()).to(device)


def corruption_B(u: torch.Tensor, Kf: torch.Tensor, coeff: float) -> torch.Tensor:
    return (coeff / Kf.shape[1]) * torch.outer(u, Kf.sum(1))
