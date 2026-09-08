"""Recovery attacks. Generate only; scoring lives in scoring.py."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from omegaconf import OmegaConf

from evals.metrics.utils import stop_sequences_criteria
from open_unlearning_plugins.leakru.scoring import (
    FOCUS_ON_KEY_REPEATS,
    PROBAB_TEMPERATURE,
    PROBAB_TOP_P,
    PROBAB_TRIALS,
    extract_key_tokens,
    focus_on_key_prompt,
)

logger = logging.getLogger("evaluator")


def _generation_dict(generation_args: Any) -> dict:
    if generation_args is None:
        return {"do_sample": False, "max_new_tokens": 200, "use_cache": True}
    if isinstance(generation_args, dict):
        return dict(generation_args)
    return OmegaConf.to_container(generation_args, resolve=True)


def generate_texts(model, tokenizer, batch, generation_args) -> list[str]:
    import torch

    batch = {k: v.to(model.device) for k, v in batch.items() if torch.is_tensor(v)}
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    gen_args = _generation_dict(generation_args)
    stopwords = gen_args.pop("stopwords", None)
    if stopwords is not None:
        gen_args["stopping_criteria"] = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **gen_args,
        pad_token_id=pad_id,
    )
    texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    cuts = [tokenizer.decode([tokenizer.eos_token_id])]
    if stopwords:
        cuts = cuts + list(stopwords)
    cleaned = []
    for raw in texts:
        text = raw
        for word in cuts:
            if word and word in text:
                text = text.split(word)[0]
        cleaned.append(text.strip())
    return cleaned


def greedy_generate(model, tokenizer, batch, generation_args) -> list[str]:
    gen_args = _generation_dict(generation_args)
    gen_args["do_sample"] = False
    gen_args.pop("temperature", None)
    gen_args.pop("top_p", None)
    return generate_texts(model, tokenizer, batch, gen_args)


def probab_generate(
    model,
    tokenizer,
    batch,
    generation_args,
    temperature: float = PROBAB_TEMPERATURE,
    top_p: float = PROBAB_TOP_P,
    num_trials: int = PROBAB_TRIALS,
) -> list[list[str]]:
    """Return num_trials decoded strings per item in the batch."""
    gen_args = _generation_dict(generation_args)
    gen_args["do_sample"] = True
    gen_args["temperature"] = temperature
    gen_args["top_p"] = top_p
    trials: list[list[str]] = [[] for _ in range(batch["input_ids"].shape[0])]
    for _ in range(num_trials):
        texts = generate_texts(model, tokenizer, batch, gen_args)
        for i, text in enumerate(texts):
            trials[i].append(text)
    return trials


def focus_on_key_generate(
    model,
    tokenizer,
    batch,
    generation_args,
    questions: Sequence[str],
    key_tokens_list: Sequence[Sequence[str] | None] | None = None,
    template_args=None,
    answers: Sequence[str] | None = None,
    repeats: int = FOCUS_ON_KEY_REPEATS,
) -> list[str]:
    """Rebuild prompts with repeated key tokens, then greedy-generate."""
    from data.utils import preprocess_chat_instance

    if template_args is None:
        logger.warning("FocusOnKey missing template_args; falling back to greedy on original batch")
        return greedy_generate(model, tokenizer, batch, generation_args)

    keyed_ids = []
    keyed_mask = []
    for i, question in enumerate(questions):
        provided = None
        if key_tokens_list is not None:
            provided = key_tokens_list[i]
        tokens = extract_key_tokens(question, provided)
        prompt = focus_on_key_prompt(question, tokens, repeats=repeats)
        answer = answers[i] if answers is not None else ""
        tok = preprocess_chat_instance(
            tokenizer,
            template_args,
            [prompt],
            [answer],
            max_length=512,
            predict_with_generate=True,
        )
        keyed_ids.append(tok["input_ids"])
        keyed_mask.append(tok["attention_mask"])

    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_len = max(t.numel() if hasattr(t, "numel") else len(t) for t in keyed_ids)
    input_ids = []
    attn = []
    for ids, mask in zip(keyed_ids, keyed_mask):
        ids = ids if hasattr(ids, "tolist") else torch.tensor(ids)
        mask = mask if hasattr(mask, "tolist") else torch.tensor(mask)
        pad_n = max_len - ids.shape[0]
        if pad_n > 0:
            ids = torch.nn.functional.pad(ids, (pad_n, 0), value=pad_id)
            mask = torch.nn.functional.pad(mask, (pad_n, 0), value=0)
        input_ids.append(ids)
        attn.append(mask)
    new_batch = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
    }
    return greedy_generate(model, tokenizer, new_batch, generation_args)


def prepare_int4_model(model):
    """Reload the same checkpoint in nf4. Needs local weights; do not call in unit tests."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    name = getattr(model, "name_or_path", None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )
    if not name:
        raise RuntimeError(
            "Quantization attack cannot resolve a checkpoint path on the model; "
            "set model.model_args.pretrained_model_name_or_path."
        )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("Loading INT4 copy from %s for LeakRU quantization attack", name)
    return AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, device_map="auto"
    )


def quantization_generate(model, tokenizer, batch, generation_args) -> list[str]:
    """Greedy generate with the provided model (caller should pass an INT4 copy)."""
    return greedy_generate(model, tokenizer, batch, generation_args)
