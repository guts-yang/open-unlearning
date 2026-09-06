"""GROM UnlearnTrainer: one-shot closed-form edit, no HF optimizer loop.

Pin Batorskq/GROM@6dd9591. Not EvoMU.
"""

from __future__ import annotations

import logging

import torch

from trainer.unlearn.base import UnlearnTrainer

from open_unlearning_plugins.grom.edit import (
    EditConfig,
    apply_grom,
    save_checkpoint,
    untie_lm_head,
)

logger = logging.getLogger(__name__)


class GROM(UnlearnTrainer):
    def __init__(
        self,
        *args,
        name: str = "grom",
        kind: str = "tofu",
        template: str = "llama3",
        forget_split: str = "",
        retain_splits=None,
        forget_jsonl: str = "",
        retain_jsonl: str = "",
        n_docs: int = 2000,
        win: int = 400,
        layers=None,
        target: str = "suppress",
        strength: float = 0.0,
        w_r: float = 1.0,
        rho: float = 0.03,
        forget_cols: int = 20000,
        retain_cols: int = 40000,
        batch_size: int = 16,
        grom_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grom_name = name
        self.kind = kind
        self.template = template
        self.forget_split = forget_split
        self.retain_splits = list(retain_splits or [])
        self.forget_jsonl = forget_jsonl
        self.retain_jsonl = retain_jsonl
        self.n_docs = n_docs
        self.win = win
        self.layers = list(layers or [])
        self.target = target
        self.strength = strength
        self.w_r = w_r
        self.rho = rho
        self.forget_cols = forget_cols
        self.retain_cols = retain_cols
        self.batch_size = batch_size
        self.grom_seed = grom_seed

    def _tokenizer(self):
        return getattr(self, "processing_class", None) or getattr(
            self, "tokenizer", None
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        tok = self._tokenizer()
        model = self.model
        model.eval()
        untie_lm_head(model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("GROM running on CPU; paper timings assume H100.")
        model.to(device)
        pretrained = getattr(model.config, "_name_or_path", "") or ""
        tok_name = getattr(tok, "name_or_path", None) or pretrained
        cfg = EditConfig(
            name=self.grom_name,
            base=pretrained,
            tokenizer=tok_name or pretrained,
            kind=self.kind,
            template=self.template,
            forget_split=self.forget_split,
            retain_splits=self.retain_splits,
            forget_jsonl=self.forget_jsonl,
            retain_jsonl=self.retain_jsonl,
            n_docs=self.n_docs,
            win=self.win,
            layers=self.layers,
            target=self.target,
            strength=self.strength,
            w_r=self.w_r,
            rho=self.rho,
            forget_cols=self.forget_cols,
            retain_cols=self.retain_cols,
            batch_size=self.batch_size,
            seed=self.grom_seed,
            device=device,
        )
        logger.info(
            "GROM edit name=%s layers=%s target=%s strength=%s w_r=%s rho=%s",
            cfg.name,
            cfg.layers,
            cfg.target,
            cfg.strength,
            cfg.w_r,
            cfg.rho,
        )
        meta = apply_grom(cfg, model, tok)
        save_checkpoint(model, tok, self.args.output_dir, meta=meta)
        self.model = model
        return None
