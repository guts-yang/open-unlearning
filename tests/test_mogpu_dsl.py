import json
from types import SimpleNamespace

import torch

from trainer.unlearn.mogpu import MOGPU
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def spec():
    return CandidateSpec.from_dict(
        {
            "schema_version": 1,
            "atoms": ["RetainDrift", "EraseResidual"],
            "weights": [2.0, 1.0],
            "thresholds": {"kappa": 0.3, "tau": 0.05},
            "approved_variants": {"EraseResidual": "softplus", "RetainDrift": "huber"},
        }
    )


def test_equivalent_candidates_have_stable_hash():
    equivalent = CandidateSpec.from_dict(
        {
            "weights": [1.0, 2.0],
            "atoms": ["EraseResidual", "RetainDrift"],
            "thresholds": {"tau": 0.05, "kappa": 0.3},
            "approved_variants": {"RetainDrift": "huber", "EraseResidual": "softplus"},
        }
    )
    assert spec().ast_hash == equivalent.ast_hash
    assert json.loads(spec().canonical_json()) == equivalent.to_dict()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16)

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_fixed_candidate_trainer_smoke_forward_backward():
    model, reference = TinyModel(), TinyModel()
    trainer = object.__new__(MOGPU)
    trainer.candidate = validate_candidate(spec())
    trainer.ref_model = reference
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3),
        "labels": torch.tensor([[-100, 2, 3]]),
    }
    loss = MOGPU.compute_loss(trainer, model, {"forget": batch, "retain": batch})
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
