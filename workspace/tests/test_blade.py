from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from open_unlearning_plugins.blade.lora_bial_losses import (  # noqa: E402
    clamped_entropy_from_logits,
)
from open_unlearning_plugins.register import registered_handler_names  # noqa: E402


def test_clamped_entropy_zero_when_uniform():
    vocab = 32
    logits = torch.zeros(2, 5, vocab, requires_grad=True)
    mask = torch.ones(2, 5)
    loss = clamped_entropy_from_logits(logits, mask, tau=0.7)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_clamped_entropy_positive_when_peaked():
    vocab = 16
    logits = torch.zeros(1, 3, vocab)
    logits[0, :, 0] = 20.0
    logits = logits.detach().requires_grad_(True)
    mask = torch.ones(1, 3)
    loss = clamped_entropy_from_logits(logits, mask, tau=0.7)
    assert float(loss) > 0
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_blade_handlers_listed():
    names = set(registered_handler_names())
    assert "LoRABiAL" in names
    assert "LoRABiALAdaptive" in names
