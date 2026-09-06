from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from open_unlearning_plugins.baldro.utils import compute_forget_group, dv_aggregate
from trainer.utils import compute_batch_nll


class ToyCausal(nn.Module):
    def __init__(self, vocab=6, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        logits = self.lm(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


def _batch(seq_ids):
    input_ids = torch.tensor(seq_ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def test_forget_group_selects_low_nll():
    torch.manual_seed(0)
    model = ToyCausal()
    # Two sequences: one aligned with random init, one adversarial ids.
    forget = _batch([[1, 2, 3, 4], [1, 1, 1, 1], [5, 4, 3, 2], [2, 2, 2, 2]])
    with torch.no_grad():
        nll, _ = compute_batch_nll(model, forget)
        counts = (forget["labels"][..., 1:] != -100).sum(-1)
        avg = nll / counts.clamp(min=1)
    selected, k, idx = compute_forget_group(model, forget, sampling_ratio=0.5)
    assert k == 2
    hardness = -avg
    expected = set(torch.topk(hardness, k=2, largest=True).indices.tolist())
    assert set(idx.tolist()) == expected
    assert selected["input_ids"].shape[0] == 2


def test_dv_large_beta_near_mean():
    losses = torch.tensor([0.2, 0.4, 0.6, 0.8])
    agg = dv_aggregate(losses, beta_dv=1e6)
    assert torch.isclose(agg, losses.mean(), rtol=0, atol=1e-4)


def test_dv_backward():
    losses = torch.tensor([0.3, 1.1, 0.7], requires_grad=True)
    agg = dv_aggregate(losses, beta_dv=2.0)
    agg.backward()
    assert losses.grad is not None
    assert torch.all(losses.grad > 0)
