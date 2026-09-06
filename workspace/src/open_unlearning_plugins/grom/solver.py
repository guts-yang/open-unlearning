"""Closed-form GROM solve (Theorem 1). Pin Batorskq/GROM@6dd9591.

P* = (w_f/s) D Xf^T A^{-1}
A = (w_r/r) Xr Xr^T + (w_f/s) Xf Xf^T + mu I
w_f is fixed at 1; mu = rho * mean(diag(A_unregularised)). Solved in float64.
"""

from __future__ import annotations

from typing import Tuple

import torch


def build_gram(
    Kf: torch.Tensor, Kr: torch.Tensor, w_r: float, rho: float, device: str
) -> Tuple[torch.Tensor, float]:
    Kf = Kf.double().to(device)
    Kr = Kr.double().to(device)
    s, r, n = Kf.shape[1], Kr.shape[1], Kf.shape[0]
    A = w_r * (Kr @ Kr.T) / r + (Kf @ Kf.T) / s
    mu = rho * (A.diagonal().sum() / n).item()
    A = A + mu * torch.eye(n, dtype=torch.float64, device=device)
    return A, mu


def solve_update(
    Kf: torch.Tensor,
    Kr: torch.Tensor,
    w_r: float,
    rho: float,
    device: str,
    D: torch.Tensor = None,
    B: torch.Tensor = None,
) -> torch.Tensor:
    if (D is None) == (B is None):
        raise ValueError("pass exactly one of D or B")
    A, _ = build_gram(Kf, Kr, w_r, rho, device)
    if B is None:
        Kf = Kf.double().to(device)
        B = (D.double().to(device) @ Kf.T) / Kf.shape[1]
    else:
        B = B.double().to(device)
    return torch.linalg.solve(A, B.T).T
