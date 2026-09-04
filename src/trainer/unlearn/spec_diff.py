import logging

import torch
from transformers import AutoModelForCausalLM

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_batch_nll, compute_specdiff_statistics


logger = logging.getLogger(__name__)


class SpecDiff(UnlearnTrainer):
    """Optimize full-vocabulary overlap against a frozen pre-unlearning model."""

    def __init__(
        self,
        draft_model_path=None,
        lam=1.0,
        beta=0.1,
        kappa=0.3,
        tau=0.02,
        warmup_steps=1,
        chunk_size=8192,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= kappa <= 1.0:
            raise ValueError("kappa must be in [0, 1]")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        if lam < 0.0 or beta < 0.0:
            raise ValueError("lam and beta must be non-negative")
        if warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be >= 1 because q and draft share initialization"
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.lam = float(lam)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.warmup_steps = int(warmup_steps)
        self.chunk_size = int(chunk_size)
        self._last_component_log_step = None

        draft_model_path = draft_model_path or getattr(
            self.model.config, "_name_or_path", None
        )
        if not draft_model_path:
            raise ValueError("draft_model_path is required")
        self.draft_model_path = draft_model_path
        self.draft_model = self._prepare_draft_model(draft_model_path)

        target_vocab = self._vocab_size(self.model)
        draft_vocab = self._vocab_size(self.draft_model)
        if target_vocab is None or draft_vocab is None:
            logger.warning(
                "Could not read vocab size from wrapped models; skipping mismatch check"
            )
        elif target_vocab != draft_vocab:
            raise ValueError(
                f"Draft/target vocab mismatch: {draft_vocab} vs {target_vocab}"
            )

    def _prepare_draft_model(self, model_path):
        load_kwargs = {"torch_dtype": self.model.dtype}
        attention_impl = getattr(self.model.config, "_attn_implementation", None)
        if attention_impl:
            load_kwargs["attn_implementation"] = attention_impl
        draft_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
        draft_model.requires_grad_(False)
        draft_model.eval()
        draft_model.to(self.accelerator.device)
        if self.is_deepspeed_enabled:
            draft_model = self._prepare_deepspeed(draft_model)
        else:
            draft_model = self.accelerator.prepare_model(
                draft_model, evaluation_mode=True
            )
        return draft_model

    @staticmethod
    def _unwrap(model):
        unwrapped = model
        for attr in ("module", "model"):
            inner = getattr(unwrapped, attr, None)
            if inner is not None and inner is not unwrapped:
                unwrapped = inner
        return unwrapped

    def _vocab_size(self, model):
        if self.processing_class is not None:
            return len(self.processing_class)
        unwrapped = self._unwrap(model)
        config = getattr(unwrapped, "config", None)
        if config is not None and getattr(config, "vocab_size", None):
            return int(config.vocab_size)
        embeddings = getattr(unwrapped, "get_input_embeddings", None)
        if callable(embeddings):
            weight = embeddings().weight
            return int(weight.shape[0])
        return None

    @staticmethod
    def _model_inputs(inputs):
        return {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "labels")
            if key in inputs
        }

    @staticmethod
    def _mean_sequence_nll(model, inputs):
        sequence_nll, outputs = compute_batch_nll(model, inputs)
        token_counts = inputs["labels"][..., 1:].ne(-100).sum(dim=-1)
        if bool((token_counts == 0).any()):
            raise ValueError("Every SpecDiff sample must contain an answer token")
        return (sequence_nll / token_counts).mean(), outputs

    def _log_components(self, **components):
        step = int(self.state.global_step)
        if self._last_component_log_step == step:
            return
        self._last_component_log_step = step
        self.log(
            {
                f"specdiff/{name}": float(value.detach().float().cpu())
                for name, value in components.items()
            }
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._model_inputs(inputs["forget"])
        retain_inputs = self._model_inputs(inputs["retain"])

        # At q == draft, sum(min(p, q)) and KL both have zero gradient.
        # One optimizer step of GradDiff opens a semantically directed gap.
        if self.state.global_step < self.warmup_steps:
            forget_nll, forget_outputs = self._mean_sequence_nll(
                model, forget_inputs
            )
            retain_nll, _ = self._mean_sequence_nll(model, retain_inputs)
            loss = -forget_nll + self.lam * retain_nll
            self._log_components(
                warmup_loss=loss,
                warmup_forget_nll=forget_nll,
                warmup_retain_nll=retain_nll,
            )
            return (loss, forget_outputs) if return_outputs else loss

        forget_forward = {
            key: forget_inputs[key] for key in ("input_ids", "attention_mask")
        }
        retain_forward = {
            key: retain_inputs[key] for key in ("input_ids", "attention_mask")
        }
        forget_outputs = model(**forget_forward)
        retain_outputs = model(**retain_forward)
        with torch.no_grad():
            draft_forget_logits = self.draft_model(**forget_forward).logits
            draft_retain_logits = self.draft_model(**retain_forward).logits

        forget_overlap, _ = compute_specdiff_statistics(
            draft_forget_logits,
            forget_outputs.logits,
            forget_inputs["labels"],
            chunk_size=self.chunk_size,
            compute_kl=False,
        )
        retain_overlap, retain_kl = compute_specdiff_statistics(
            draft_retain_logits,
            retain_outputs.logits,
            retain_inputs["labels"],
            chunk_size=self.chunk_size,
            compute_kl=True,
        )

        forget_loss = forget_overlap.clamp(min=1.0 - self.kappa).mean()
        retain_loss = (1.0 - retain_overlap).clamp(min=self.tau).mean()
        kl_loss = retain_kl.mean()
        loss = forget_loss + self.lam * retain_loss + self.beta * kl_loss
        self._log_components(
            loss=loss,
            forget_overlap=forget_overlap.mean(),
            retain_overlap=retain_overlap.mean(),
            forget_loss=forget_loss,
            retain_loss=retain_loss,
            retain_kl=kl_loss,
        )
        return (loss, forget_outputs) if return_outputs else loss
