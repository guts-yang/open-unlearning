import numpy as np
import torch

from evals.specgap import (
    DraftLogitsCache,
    bootstrap_mean_ci,
    cohens_d,
    min_distribution_overlap,
    prepare_probe_item,
)


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(
        self,
        text,
        add_special_tokens=True,
        max_length=None,
        truncation=False,
    ):
        ids = [ord(character) + 10 for character in text]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


TEMPLATE = {
    "apply_chat_template": False,
    "user_start_tag": "<u>",
    "user_end_tag": "</u>",
    "asst_start_tag": "<a>",
    "asst_end_tag": "</a>",
}


def test_identical_distributions_have_zero_gap():
    logits = torch.randn(3, 11)
    overlap = min_distribution_overlap(logits, logits.clone(), chunk_size=4)
    assert torch.allclose(overlap, torch.ones_like(overlap), atol=1e-6)


def test_disjoint_distributions_have_unit_gap():
    draft = torch.tensor([[50.0, -50.0], [-50.0, 50.0]])
    target = torch.tensor([[-50.0, 50.0], [50.0, -50.0]])
    overlap = min_distribution_overlap(draft, target, chunk_size=1)
    assert torch.all(overlap < 1e-6)


def test_chunked_overlap_matches_full_softmax():
    generator = torch.Generator().manual_seed(7)
    draft = torch.randn(5, 17, generator=generator)
    target = torch.randn(5, 17, generator=generator)
    expected = torch.minimum(draft.softmax(dim=-1), target.softmax(dim=-1)).sum(dim=-1)
    actual = min_distribution_overlap(draft, target, chunk_size=3)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_answer_mask_is_shifted_to_prediction_positions():
    tokenizer = CharacterTokenizer()
    item = prepare_probe_item(tokenizer, TEMPLATE, "Q", "AB", max_length=128)
    prompt_text = "<u>Q</u><a>"
    prompt_tokens = 1 + len(prompt_text)

    assert item["prediction_mask"].sum().item() == 3  # A, B, appended EOS
    assert item["prediction_mask"].nonzero()[0].item() == prompt_tokens - 1


def test_bootstrap_is_reproducible():
    values = [0.1, 0.2, 0.4, 0.8]
    assert bootstrap_mean_ci(values, seed=9) == bootstrap_mean_ci(values, seed=9)


def test_cohens_d_direction_and_zero_variance():
    assert cohens_d([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]) > 0
    assert cohens_d([0.2, 0.2], [0.2, 0.2]) == 0
    assert np.isfinite(cohens_d([0.3, 0.3], [0.2, 0.2]))


def test_draft_cache_round_trip(tmp_path):
    metadata = {"split": "forget10", "indices": [0]}
    cache = DraftLogitsCache(tmp_path, metadata)
    cache.initialize()
    logits = torch.randn(2, 7)
    cache.save(0, logits)

    restored = cache.load(0)
    assert restored.dtype == torch.bfloat16
    assert torch.allclose(restored.float(), logits, atol=0.01)
