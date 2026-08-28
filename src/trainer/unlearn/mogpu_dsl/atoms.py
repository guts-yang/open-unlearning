"""The complete whitelist of MOGP-U mechanism atoms."""

from __future__ import annotations

import torch
from torch.nn import functional


def erase_residual(
    delta_forget: torch.Tensor, kappa: float, temperature: float
) -> torch.Tensor:
    """Penalize residual forget likelihood; derivative is non-negative."""
    return functional.softplus((delta_forget + kappa) / temperature)


def retain_drift(delta_retain: torch.Tensor, tau: float) -> torch.Tensor:
    """Huber drift penalty; its derivative has the sign of delta_retain."""
    return functional.huber_loss(
        delta_retain, torch.zeros_like(delta_retain), delta=tau, reduction="none"
    )


def selective_margin(
    delta_forget: torch.Tensor,
    delta_retain: torch.Tensor,
    kappa: float,
    temperature: float,
    epsilon: float,
) -> torch.Tensor:
    """Require -Delta_f to exceed the absolute retain drift by kappa."""
    retain_magnitude = torch.sqrt(delta_retain.square() + epsilon * epsilon)
    return functional.softplus((retain_magnitude + delta_forget + kappa) / temperature)
