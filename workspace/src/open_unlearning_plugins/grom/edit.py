"""Sequential closed-form down_proj edits. Pin Batorskq/GROM@6dd9591."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from open_unlearning_plugins.grom.data import (
    corpus_windows,
    flat_counts,
    tofu_pairs,
    token_counts,
    tokenize_pairs,
)
from open_unlearning_plugins.grom.keys import forward_collect
from open_unlearning_plugins.grom.solver import solve_update
from open_unlearning_plugins.grom.targets import (
    corruption_B,
    corruption_direction,
    specificity_alpha,
    suppression_D,
    unit_unembeddings,
)

logger = logging.getLogger(__name__)

GROM_PIN = "Batorskq/GROM@6dd9591"


@dataclass
class EditConfig:
    name: str
    base: str
    tokenizer: str
    kind: str
    template: str = "llama3"
    forget_split: str = ""
    retain_splits: List[str] = field(default_factory=list)
    forget_jsonl: str = ""
    retain_jsonl: str = ""
    n_docs: int = 2000
    win: int = 400
    layers: List[int] = field(default_factory=list)
    target: str = "suppress"
    strength: float = 0.0
    w_r: float = 1.0
    rho: float = 0.03
    forget_cols: int = 20000
    retain_cols: int = 40000
    batch_size: int = 16
    seed: int = 0
    device: str = "cuda"


def untie_lm_head(model) -> None:
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False


def build_data(cfg: EditConfig, tok, vocab_size: int):
    if cfg.kind == "tofu":
        forget = tofu_pairs(cfg.forget_split)
        retain = []
        for split in cfg.retain_splits:
            part = tofu_pairs(split)
            retain += part
            logger.info("retain anchor %s: %s pairs", split, len(part))
        fi, fl, fg = tokenize_pairs(tok, forget, True, cfg.template)
        ri, rl, _ = tokenize_pairs(tok, retain, False, cfg.template)
        retain_counts = token_counts(tok, [a for _, a in retain], vocab_size)
    elif cfg.kind == "corpus":
        fi, fl, fg = corpus_windows(
            tok, cfg.forget_jsonl, cfg.n_docs, cfg.win, cfg.forget_cols
        )
        ri, rl, rg = corpus_windows(
            tok, cfg.retain_jsonl, cfg.n_docs, cfg.win, cfg.retain_cols
        )
        retain_counts = flat_counts(rg, vocab_size)
    else:
        raise ValueError(f"unknown data kind: {cfg.kind}")

    alpha = None
    if cfg.target == "suppress":
        alpha = specificity_alpha(flat_counts(fg, vocab_size), retain_counts).to(
            cfg.device
        )
    return fi, fl, fg, ri, rl, alpha


def apply_grom(cfg: EditConfig, model, tok) -> dict:
    t0 = time.time()
    vocab_size, _d_model = model.lm_head.weight.shape
    uhat = unit_unembeddings(model)
    fi, fl, fg, ri, rl, alpha = build_data(cfg, tok, vocab_size)
    flat_gold = torch.tensor([g for gs in fg for g in gs]).to(cfg.device)
    layer_logs = []

    for layer in cfg.layers:
        down_proj = model.model.layers[layer].mlp.down_proj
        Kf, idx = forward_collect(
            model,
            tok,
            fi,
            fl,
            cfg.device,
            module=down_proj,
            bs=cfg.batch_size,
            max_cols=cfg.forget_cols,
        )
        Kr, _ = forward_collect(
            model,
            tok,
            ri,
            rl,
            cfg.device,
            module=down_proj,
            bs=cfg.batch_size,
            max_cols=cfg.retain_cols,
        )
        if cfg.target == "suppress":
            gold = flat_gold[idx.to(cfg.device)] if idx is not None else flat_gold
            D = suppression_D(uhat, alpha, gold, cfg.strength)
            P = solve_update(Kf, Kr, cfg.w_r, cfg.rho, cfg.device, D=D)
        else:
            u = corruption_direction(down_proj.weight.shape[0], cfg.seed, cfg.device)
            B = corruption_B(u, Kf.double().to(cfg.device), cfg.strength)
            P = solve_update(Kf, Kr, cfg.w_r, cfg.rho, cfg.device, B=B)
        down_proj.weight.data.add_(P.to(down_proj.weight.dtype))
        layer_logs.append(
            {
                "layer": layer,
                "keys": list(Kf.shape),
                "P_fro": float(P.float().norm()),
            }
        )
        logger.info(
            "edited layer %s keys=%s ||P||_F=%.3f",
            layer,
            tuple(Kf.shape),
            float(P.float().norm()),
        )

    elapsed = time.time() - t0
    return {
        "pin": GROM_PIN,
        "name": cfg.name,
        "layers": list(cfg.layers),
        "target": cfg.target,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "layer_logs": layer_logs,
    }


def save_checkpoint(model, tok, out_dir: str, meta: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    if tok is not None:
        tok.save_pretrained(out_dir)
    if meta is not None:
        with open(os.path.join(out_dir, "grom_edit.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
