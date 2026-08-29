"""Toy synthetic tests for the ``src.esmu.subspace`` module (Stage 3, G1' gate code).

WARNING: every number in this file comes from SYNTHETIC toy data, NOT from a real
checkpoint. Per the task's data-integrity rules, synthetic data must be clearly
labelled as toy — these tests only validate the linear-algebra contract of the
four interfaces, they produce no paper numbers.

Covered by these tests:
    1. ``accumulate_gram`` matches the manual outer-product sum ``Σ_k g_k g_kᵀ``
       (K probes, each with its own forward+backward, no batched materialization).
    2. ``param_filter`` selects a parameter subset (injection point of the
       undecided ∇s_k parameter scope, OPEN_DESIGN_DECISIONS #2).
    3. A custom ``scalar_fn`` drives the Gram (injection point of the undecided
       s_k definition, OPEN_DESIGN_DECISIONS #1).
    4. ``effective_rank`` counts eigenvalues above ``eps * λ_max``, with edge
       cases (identity, degenerate diagonal, all-zeros) and arg validation.
    5. ``whitening_sqrt_inv`` satisfies ``G^{-1/2} G G^{-1/2} ≈ I``, is
       symmetric, and stays finite on a near-singular Gram (clamp prevents
       division by zero).
    6. ``sample_direction`` is reproducible with an injected ``torch.Generator``,
       and the whitened perturbation ``Δ = Jᵀ c`` has isotropic expected norm
       (``E‖Δ‖² ≈ K``), i.e. it explores S without a Gram-induced bias.

References (主方案_v3_ES-MU_2026-08-28.md):
    - §7: S = span{∇_θ s_k}, G_ij = ⟨∇s_i, ∇s_j⟩.
    - §8: Δ = ∇_θ⟨G^{-1/2} c, s(θ)⟩ stays inside S; calling code must backprop
      ``Δ = Σ_k c_k·∇_θ s_k`` on the fly (never cache the K basis vectors).
    - §10: G1' requires K_eff ≤ 256.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from esmu.subspace import (
    _default_scalar_nll_sum,
    accumulate_gram,
    effective_rank,
    sample_direction,
    whitening_sqrt_inv,
)


class _ToyCausalLM(nn.Module):
    """Tiny SYNTHETIC CausalLM whose outputs expose ``.logits`` like a HF model."""

    def __init__(self, n_vocab: int = 7, n_emb: int = 2, n_hidden: int = 3):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, n_emb)
        self.fc1 = nn.Linear(n_emb, n_hidden, bias=True)
        self.fc2 = nn.Linear(n_hidden, n_vocab, bias=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels  # keep the signature compatible with **batch
        hidden = F.relu(self.fc1(self.embed(input_ids)))
        return SimpleNamespace(logits=self.fc2(hidden))


def _probe_batches(n_probes: int, seq_len: int = 4, n_vocab: int = 7, seed: int = 0):
    """Build SYNTHETIC probe batches with the DataCollatorForSupervisedDataset keys."""
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_probes):
        input_ids = torch.randint(0, n_vocab, (1, seq_len), generator=gen)
        labels = input_ids.clone()
        labels[:, 0] = -100  # exercise the ignore_index path
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
        )
    return batches


def _jacobian_rows(model, batches, scalar_fn=_default_scalar_nll_sum):
    """Test-only helper: K × d matrix whose rows are vec(∇s_k). Holding K tiny
    d-vectors here is fine; the module's O(d)+O(K²) memory constraint targets
    the 1B/A800 runtime path, not toy tests."""
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for batch in batches:
        g = torch.autograd.grad(scalar_fn(model, batch), params, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (
                        gi.reshape(-1)
                        if gi is not None
                        else torch.zeros(p.numel(), device=p.device)
                    )
                    for gi, p in zip(g, params)
                ]
            )
        )
    return torch.stack(rows)


def test_accumulate_gram_matches_manual_outer_product() -> None:
    torch.manual_seed(0)
    model = _ToyCausalLM()
    batches = _probe_batches(n_probes=3)

    gram = accumulate_gram(model, batches)
    expected = _jacobian_rows(model, batches)
    expected_gram = expected @ expected.T

    assert gram.shape == (3, 3)
    assert torch.allclose(gram, expected_gram, atol=1e-5)


def test_accumulate_gram_param_filter_subset() -> None:
    torch.manual_seed(0)
    model = _ToyCausalLM()
    batches = _probe_batches(n_probes=2)

    gram = accumulate_gram(
        model, batches, param_filter=lambda name, param: name.startswith("fc2")
    )
    params = [p for n, p in model.named_parameters() if n.startswith("fc2")]
    rows = []
    for batch in batches:
        s = _default_scalar_nll_sum(model, batch)
        g = torch.autograd.grad(s, params)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    expected_gram = torch.stack(rows) @ torch.stack(rows).T

    assert torch.allclose(gram, expected_gram, atol=1e-5)


def test_accumulate_gram_custom_scalar_fn() -> None:
    torch.manual_seed(0)
    model = _ToyCausalLM()
    batches = _probe_batches(n_probes=2)

    def s_k(model, batch):
        hidden = F.relu(model.fc1(model.embed(batch["input_ids"])))
        return hidden.sum()

    gram = accumulate_gram(model, batches, scalar_fn=s_k)
    expected_gram = _jacobian_rows(model, batches, scalar_fn=s_k)
    assert torch.allclose(gram, expected_gram @ expected_gram.T, atol=1e-5)


def test_accumulate_gram_rejects_non_scalar_output() -> None:
    model = _ToyCausalLM()
    batches = _probe_batches(n_probes=1)

    def vector_scalar(model, batch):
        del model, batch
        return torch.zeros(3)

    with pytest.raises(ValueError, match="0-dim"):
        accumulate_gram(model, batches, scalar_fn=vector_scalar)


def test_accumulate_gram_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        accumulate_gram(_ToyCausalLM(), [])


def test_effective_rank_identity() -> None:
    assert effective_rank(torch.eye(5), eps=1e-6) == 5


def test_effective_rank_degenerate_diagonal() -> None:
    G = torch.diag(torch.tensor([1.0, 1e-8, 1e-8, 0.0, 0.0]))
    assert effective_rank(G, eps=1e-3) == 1


def test_effective_rank_all_zeros() -> None:
    assert effective_rank(torch.zeros(4, 4), eps=1e-3) == 0


def test_effective_rank_arg_validation() -> None:
    with pytest.raises(ValueError, match="eps"):
        effective_rank(torch.eye(2), eps=0.0)
    with pytest.raises(ValueError, match="square"):
        effective_rank(torch.ones(2, 3), eps=1e-3)


def test_whitening_sqrt_inv_inverse_property() -> None:
    torch.manual_seed(0)
    A = torch.randn(4, 6)
    G = A @ A.T + 1e-6 * torch.eye(4)

    W = whitening_sqrt_inv(G, eps=1e-4)

    assert torch.allclose(W @ G @ W, torch.eye(4), atol=5e-3)
    assert torch.allclose(W, W.T, atol=1e-6)  # symmetric


def test_whitening_sqrt_inv_near_singular_finite() -> None:
    G = torch.diag(torch.tensor([1e-10, 1.0, 0.0]))
    W = whitening_sqrt_inv(G, eps=1e-6)
    assert torch.isfinite(W).all()


def test_whitening_sqrt_inv_arg_validation() -> None:
    with pytest.raises(ValueError, match="eps"):
        whitening_sqrt_inv(torch.eye(2), eps=0.0)
    with pytest.raises(ValueError, match="square"):
        whitening_sqrt_inv(torch.ones(2, 3), eps=1e-3)


def test_sample_direction_reproducible_with_generator() -> None:
    identity = torch.eye(4)
    c1 = sample_direction(identity, rng=torch.Generator().manual_seed(0))
    c2 = sample_direction(identity, rng=torch.Generator().manual_seed(0))

    assert c1.shape == (4,)
    assert torch.equal(c1, c2)
    # G^{-1/2} = I ⇒ c == z; verify against the underlying standard normal draw
    z = torch.randn(4, generator=torch.Generator().manual_seed(0))
    assert torch.equal(c1, z)


def test_sample_direction_whitened_isotropic_norm() -> None:
    # Whitening removes the Gram-induced bias: E‖Jᵀc‖² ≈ K instead of tr(G).
    torch.manual_seed(0)
    model = _ToyCausalLM(n_vocab=9, n_emb=3, n_hidden=4)
    batches = _probe_batches(n_probes=3, n_vocab=9)
    J = _jacobian_rows(model, batches)
    W = whitening_sqrt_inv(J @ J.T, eps=1e-4)
    K = J.shape[0]

    norms_sq = []
    for _ in range(300):
        c = sample_direction(W, rng=None)
        delta = J.T @ c  # Δ = Σ_k c_k·∇s_k, computed on the fly by the caller
        norms_sq.append(float(delta.square().sum()))
    mean_norm_sq = sum(norms_sq) / len(norms_sq)

    # E‖Jᵀc‖² = E‖z‖² = K (whitened); without whitening it would be tr(G) ≈ 88.
    assert abs(mean_norm_sq - K) < 1.0


def test_sample_direction_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        sample_direction(torch.ones(2, 3))
