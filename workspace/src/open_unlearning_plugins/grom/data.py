"""TOFU QA pairs and corpus windows. Pin Batorskq/GROM@6dd9591."""

from __future__ import annotations

import json
from typing import List, Tuple

import torch

SYSTEM_PROMPT = "You are a helpful assistant."
DATE_STRING = "10 Apr 2025"


def tofu_pairs(split: str) -> List[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", split)["train"]
    return [(e["question"], e["answer"]) for e in ds]


def build_chat(tok, question: str, answer: str, template: str = "llama3"):
    if template == "llama2":
        prompt = "[INST] " + question + " [/INST]"
        pids = tok(prompt, add_special_tokens=True).input_ids
        ids = tok(prompt + answer, add_special_tokens=True).input_ids
        if ids[-1] != tok.eos_token_id:
            ids = ids + [tok.eos_token_id]
        return ids, len(pids)

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    ids = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, date_string=DATE_STRING
    )
    pids = tok.apply_chat_template(
        chat[:-1], tokenize=True, add_generation_prompt=True, date_string=DATE_STRING
    )
    if ids[-1] != tok.eos_token_id:
        ids = ids + [tok.eos_token_id]
    return ids, len(pids)


def tokenize_pairs(tok, pairs, want_gold: bool, template: str = "llama3"):
    ids_list, lp_list, gold_list = [], [], []
    for q, a in pairs:
        ids, lp = build_chat(tok, q, a, template)
        ids_list.append(ids)
        lp_list.append(lp)
        if want_gold:
            gold_list.append(
                [ids[p + 1] for p in range(lp - 1, len(ids) - 1)] or [ids[-1]]
            )
    return ids_list, lp_list, (gold_list if want_gold else None)


def corpus_windows(tok, path: str, n_docs: int, win: int, max_cols: int):
    ids_list, gold_list = [], []
    need = max_cols
    with open(path) as handle:
        for li, line in enumerate(handle):
            if li >= n_docs or need <= 0:
                break
            obj = json.loads(line)
            if isinstance(obj, dict):
                txt = obj.get("text") or obj.get("response") or ""
            else:
                txt = obj
            ids = tok(
                txt, add_special_tokens=True, truncation=True, max_length=4096
            ).input_ids
            for start in range(0, len(ids) - 1, win):
                w = ids[start : start + win + 1]
                if len(w) < 8:
                    continue
                ids_list.append(w)
                gold_list.append(w[1:])
                need -= len(w) - 1
                if need <= 0:
                    break
    return ids_list, [1] * len(ids_list), gold_list


def token_counts(tok, texts, vocab_size: int) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.double)
    for text in texts:
        for t in tok(text, add_special_tokens=False).input_ids:
            if t < vocab_size:
                counts[t] += 1
    return counts


def flat_counts(gold_lists, vocab_size: int) -> torch.Tensor:
    flat = torch.tensor([g for gs in gold_lists for g in gs])
    return torch.bincount(flat, minlength=vocab_size).double()
