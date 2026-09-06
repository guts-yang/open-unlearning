from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from open_unlearning_plugins.grom.solver import build_gram, solve_update  # noqa: E402
from open_unlearning_plugins.grom.targets import (  # noqa: E402
    corruption_B,
    corruption_direction,
    specificity_alpha,
    suppression_D,
)
from open_unlearning_plugins.register import registered_handler_names  # noqa: E402


def _objective(P, Kf, Kr, D, w_r, mu):
    s = Kf.shape[1]
    r = Kr.shape[1]
    retain = (w_r / r) * (P @ Kr).pow(2).sum()
    forget = ((P @ Kf - D).pow(2).sum()) / s
    ridge = mu * P.pow(2).sum()
    return retain + forget + ridge


def test_solve_update_stationarity():
    torch.manual_seed(0)
    n, s, r, d_out = 8, 5, 6, 4
    Kf = torch.randn(n, s)
    Kr = torch.randn(n, r)
    D = torch.randn(d_out, s)
    device = "cpu"
    w_r, rho = 2.0, 0.03
    P = solve_update(Kf, Kr, w_r, rho, device, D=D)
    A, mu = build_gram(Kf, Kr, w_r, rho, device)
    B = D.double() @ Kf.double().T / s
    assert torch.allclose(P @ A, B, atol=1e-8, rtol=1e-6)
    base = _objective(P, Kf.double(), Kr.double(), D.double(), w_r, mu)
    noise = torch.randn_like(P) * 0.05
    bumped = _objective(P + noise, Kf.double(), Kr.double(), D.double(), w_r, mu)
    assert float(bumped) > float(base)


def test_rho_scales_ridge():
    torch.manual_seed(1)
    Kf = torch.randn(5, 3)
    Kr = torch.randn(5, 4)
    A0, mu0 = build_gram(Kf, Kr, w_r=1.0, rho=0.0, device="cpu")
    _A, mu = build_gram(Kf, Kr, w_r=1.0, rho=0.03, device="cpu")
    mean_diag = A0.diagonal().mean().item()
    assert abs(mu - 0.03 * mean_diag) < 1e-12
    assert mu0 == 0.0


def test_suppress_and_rmu_shapes():
    vocab, d_model, s, n = 10, 6, 4, 5
    uhat = torch.nn.functional.normalize(torch.randn(vocab, d_model), dim=1)
    alpha = torch.rand(vocab)
    gold = torch.tensor([1, 2, 3, 1])
    D = suppression_D(uhat, alpha, gold, beta=2.0)
    assert D.shape == (d_model, s)
    u = corruption_direction(d_model, seed=0, device="cpu")
    Kf = torch.randn(n, s)
    B = corruption_B(u, Kf.double(), coeff=45.0)
    assert B.shape == (d_model, n)
    ff = torch.ones(vocab)
    rf = torch.ones(vocab)
    a = specificity_alpha(ff, rf)
    assert torch.all(a >= 0) and torch.all(a <= 1)


def test_grom_handler_listed():
    assert "GROM" in set(registered_handler_names())
