import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evals.metrics.utils import aggregate_to_1D
from evals.metrics.harmonic import harmonic_mean, hm_2d, hm_3d
from evals.metrics.base import unlearning_metric


def _precompute_agg_values(kwargs):
    return [result["agg_value"] for _, result in kwargs["pre_compute"].items()]


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    """Harmonic mean of precomputed metric ``agg_value``s (any arity).

    Optional yaml ``n_terms`` (2 or 3) enforces the paper's 2D / 3D HM.
    """
    values = _precompute_agg_values(kwargs)
    n_terms = kwargs.get("n_terms")
    return {"agg_value": harmonic_mean(values, n=n_terms)}


@unlearning_metric(name="hm_2d")
def hm_2d_aggregate(model, **kwargs):
    """Two-term HM, e.g. Table 6 Agg = HM(Mem, Utility) or Robustness = HM(R, Q)."""
    a, b = _precompute_agg_values(kwargs)
    return {"agg_value": hm_2d(a, b)}


@unlearning_metric(name="hm_3d")
def hm_3d_aggregate(model, **kwargs):
    """Three-term HM, e.g. Table 3 Agg = HM(Mem, Priv, Utility)."""
    a, b, c = _precompute_agg_values(kwargs)
    return {"agg_value": hm_3d(a, b, c)}


@unlearning_metric(name="classifier_prob")
def classifier_prob(model, **kwargs):
    batch_size = kwargs.get("batch_size", 32)
    max_length = kwargs.get("max_length", 512)
    class_id = kwargs.get("class_id", 0)
    text_key = kwargs.get("text_key", "generation")
    classifier_model_args = kwargs["classifier_model_args"]
    classifier_tokenization_args = kwargs["classifier_tokenization_args"]
    device = kwargs.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(**classifier_tokenization_args)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        **classifier_model_args
    ).to(device)

    data = kwargs["pre_compute"]["text"]["value_by_index"]
    data_list = [
        {"text": entry[text_key], "index": int(key)} for key, entry in data.items()
    ]

    # Create DataLoader
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    scores_by_index = {}
    for batch in tqdm(dataloader):
        batch_texts = batch["text"]
        batch_indices = batch["index"].tolist()

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run the classifier
        with torch.no_grad():
            outputs = classifier(**inputs)
        # Convert logits to probabilities
        scores = F.softmax(outputs.logits, dim=-1)[:, class_id].cpu().numpy().tolist()

        # Map predictions to labels
        for idx, prob, text in zip(batch_indices, scores, batch_texts):
            # Add the prediction to the original data
            scores_by_index[idx] = {"score": prob, text_key: text}
    class_scores = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    class_scores = aggregate_to_1D(class_scores)
    return {"agg_value": np.mean(class_scores), "value_by_index": scores_by_index}
