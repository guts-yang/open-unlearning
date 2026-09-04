from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from evals.specgap import min_distribution_overlap
from trainer.unlearn.spec_diff import SpecDiff
from trainer.utils import compute_specdiff_statistics


def _batch(input_ids, labels):
    input_ids = torch.tensor(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor(labels),
    }


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=7):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.table(input_ids))


def test_training_overlap_matches_audit_and_draft_is_detached():
    generator = torch.Generator().manual_seed(3)
    draft = torch.randn(2, 4, 11, generator=generator, requires_grad=True)
    target = torch.randn(2, 4, 11, generator=generator, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3], [-100, 1, 2, 3]])

    overlap, kl = compute_specdiff_statistics(
        draft, target, labels, chunk_size=3
    )
    expected = []
    for batch_index in range(2):
        mask = labels[batch_index, 1:].ne(-100)
        token_overlap = min_distribution_overlap(
            draft[batch_index, :-1][mask],
            target[batch_index, :-1][mask],
            chunk_size=3,
        )
        expected.append(token_overlap.mean())

    assert torch.allclose(overlap, torch.stack(expected), atol=1e-6)
    (overlap.mean() + kl.mean()).backward()
    assert target.grad is not None and target.grad.abs().sum() > 0
    assert draft.grad is None


def test_overlap_losses_have_correct_gradient_directions():
    draft = torch.tensor([[[2.0, 0.0, -1.0], [1.0, 0.0, -1.0]]])
    labels = torch.tensor([[-100, 1]])

    forget_target = torch.tensor(
        [[[0.5, 1.5, -1.0], [1.0, 0.0, -1.0]]], requires_grad=True
    )
    before_f, _ = compute_specdiff_statistics(
        draft, forget_target, labels, chunk_size=2, compute_kl=False
    )
    before_f.mean().backward()
    with torch.no_grad():
        forget_target -= 0.2 * forget_target.grad
    after_f, _ = compute_specdiff_statistics(
        draft, forget_target, labels, chunk_size=2, compute_kl=False
    )
    assert after_f.item() < before_f.item()

    retain_target = torch.tensor(
        [[[0.5, 1.5, -1.0], [1.0, 0.0, -1.0]]], requires_grad=True
    )
    before_r, _ = compute_specdiff_statistics(
        draft, retain_target, labels, chunk_size=2, compute_kl=False
    )
    (1.0 - before_r).mean().backward()
    with torch.no_grad():
        retain_target -= 0.2 * retain_target.grad
    after_r, _ = compute_specdiff_statistics(
        draft, retain_target, labels, chunk_size=2, compute_kl=False
    )
    assert after_r.item() > before_r.item()


def test_masked_kl_is_q_to_draft():
    draft = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[-100, 1]])
    _, actual = compute_specdiff_statistics(
        draft, target, labels, chunk_size=1
    )

    log_p = draft[0, 0].log_softmax(dim=-1)
    log_q = target[0, 0].log_softmax(dim=-1)
    expected = (log_q.exp() * (log_q - log_p)).sum()
    assert torch.allclose(actual[0], expected, atol=1e-6)


def test_warmup_breaks_symmetry_then_specdiff_has_gradient():
    torch.manual_seed(4)
    model = TinyCausalLM()
    draft = deepcopy(model)
    draft.requires_grad_(False)

    trainer = object.__new__(SpecDiff)
    trainer.lam = 1.0
    trainer.beta = 0.1
    trainer.kappa = 0.3
    trainer.tau = 0.02
    trainer.warmup_steps = 1
    trainer.chunk_size = 3
    trainer.draft_model = draft
    trainer.state = SimpleNamespace(global_step=0)
    trainer._last_component_log_step = 0

    inputs = {
        "forget": _batch([[0, 1, 2]], [[-100, 1, 2]]),
        "retain": _batch([[3, 4, 5]], [[-100, 4, 5]]),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    warmup_loss = trainer.compute_loss(model, inputs)
    warmup_loss.backward()
    warmup_grad = sum(
        parameter.grad.abs().sum() for parameter in model.parameters()
    )
    assert warmup_grad > 0
    optimizer.step()
    optimizer.zero_grad()

    trainer.state.global_step = 1
    trainer._last_component_log_step = 1
    specdiff_loss = trainer.compute_loss(model, inputs)
    specdiff_loss.backward()
    specdiff_grad = sum(
        parameter.grad.abs().sum() for parameter in model.parameters()
    )
    assert specdiff_grad > 0
    assert all(parameter.grad is None for parameter in draft.parameters())


def test_kappa_and_tau_clamps_are_flat_after_targets():
    forget_overlap = torch.tensor([0.5], requires_grad=True)
    retain_overlap = torch.tensor([0.99], requires_grad=True)
    loss = forget_overlap.clamp(min=0.7).mean()
    loss = loss + (1.0 - retain_overlap).clamp(min=0.02).mean()
    loss.backward()
    assert forget_overlap.grad.item() == 0.0
    assert retain_overlap.grad.item() == 0.0
