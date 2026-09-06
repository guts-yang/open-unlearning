from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from open_unlearning_plugins.alter.asym_lora import (
    AsymLoRALinear,
    ihl_loss_from_logits,
    wrap_linear_modules,
)


def test_ihl_backward():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, 1, 2, 3, 4],
            [-100, 0, 1, 2, 3],
        ]
    )
    loss = ihl_loss_from_logits(logits, labels)
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_asym_lora_freeze_base_and_train_adapters():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(6, 6)
            self.other = nn.Linear(6, 6)

        def forward(self, x):
            return self.q_proj(x)

    model = Tiny()
    wrap_linear_modules(
        model,
        ["q_proj"],
        rank=2,
        n_forget_experts=2,
        q=1.5,
        high_entropy_cut=0.0,
    )
    assert isinstance(model.q_proj, AsymLoRALinear)
    assert not isinstance(model.other, AsymLoRALinear)
    x = torch.randn(3, 4, 6)
    y = model(x)
    y.sum().backward()
    assert model.q_proj.A.grad is not None
    assert model.q_proj.base.weight.grad is None
    assert model.q_proj._last_gate is not None
    assert model.q_proj._last_gate.shape[-1] == 2
