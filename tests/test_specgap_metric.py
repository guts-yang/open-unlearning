import importlib
from types import SimpleNamespace

import torch
from torch import nn

from evals.base import Evaluator


specgap_module = importlib.import_module("evals.metrics.specgap")
specgap_metric = specgap_module.specgap
_selected_positions = specgap_module._selected_positions


class FakeTokenizer:
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self):
        return {"a": 0, "b": 1, "<eos>": 2}


class FakeDataset:
    def __init__(self, size):
        self.size = size
        self.max_length = 3
        self.data = SimpleNamespace(_fingerprint=f"fake-{size}")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index % 3, 1, 2]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "labels": torch.tensor([-100, 1, 2]),
            "index": index,
        }


class FakeModel(nn.Module):
    def __init__(self, offset=0.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.offset = offset
        self.config = SimpleNamespace(vocab_size=3, _commit_hash="fake")

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, input_ids, attention_mask=None):
        base = torch.nn.functional.one_hot(input_ids, num_classes=3).float() * 3
        shift = torch.tensor([self.offset, -self.offset, 0.0])
        return SimpleNamespace(logits=base + shift)


def test_retain_selection_is_seeded_and_sorted():
    dataset = FakeDataset(20)
    first = _selected_positions(dataset, 5, seed=7)
    second = _selected_positions(dataset, 5, seed=7)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 5


def test_specgap_metric_returns_structured_summary(monkeypatch, tmp_path):
    tokenizer = FakeTokenizer()
    draft = FakeModel(offset=0.0)
    target = FakeModel(offset=0.7)
    monkeypatch.setattr(
        specgap_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        specgap_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: draft,
    )

    result = specgap_metric._metric_fn(
        target,
        data={"forget": FakeDataset(3), "retain": FakeDataset(5)},
        tokenizer=tokenizer,
        template_args={"apply_chat_template": False},
        draft_model_path="fake-draft",
        retain_n=3,
        seed=2,
        chunk_size=2,
        n_bootstrap=20,
        cache_dir=tmp_path,
    )

    aggregate = result["agg_value"]
    assert aggregate["forget"]["n"] == 3
    assert aggregate["retain"]["n"] == 3
    assert "ci95" in aggregate["forget"]
    assert "mean_difference" in aggregate
    assert "cohens_d" in aggregate

    evaluator = object.__new__(Evaluator)
    evaluator.metrics = {"specgap": specgap_metric}
    assert evaluator.summarize({"specgap": result})["specgap"] == aggregate
