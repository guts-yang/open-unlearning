"""ALTER trainer: reconstructed two-phase AsymLoRA unlearning.

Official repository has no implementation. Do not treat measured numbers as a
faithful copy of the authors' code.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from trainer.unlearn.base import UnlearnTrainer

from open_unlearning_plugins.alter.asym_lora import (
    ihl_loss_from_logits,
    iter_asym_modules,
    wrap_linear_modules,
)

logger = logging.getLogger(__name__)


class ALTER(UnlearnTrainer):
    def __init__(
        self,
        *args,
        lora_r: int = 8,
        n_forget_experts: int = 2,
        tsallis_q: float = 1.5,
        high_entropy_cut: float = 1.2,
        tau_high: float = 0.8,
        tau_low: float = 0.01,
        top_k: int = 3,
        target_modules: list | None = None,
        beta: float = 1.0,
        gamma: float = 1.0,
        lambda_entropy: float = 0.01,
        eta_a: float = 1e-5,
        eta_b: float = 1e-3,
        reconstruction_note: str = "paper-appendix-missing",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lora_r = lora_r
        self.n_forget_experts = n_forget_experts
        self.tsallis_q = tsallis_q
        self.high_entropy_cut = high_entropy_cut
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.top_k = top_k
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.beta = beta
        self.gamma_retain = gamma
        self.lambda_entropy = lambda_entropy
        self.eta_a = eta_a
        self.eta_b = eta_b
        self.reconstruction_note = reconstruction_note
        self._wrapped = False
        logger.warning(
            "ALTER is a non-official reconstruction (%s). "
            "q=%s experts=%s targets=%s",
            reconstruction_note,
            tsallis_q,
            n_forget_experts,
            self.target_modules,
        )

    def _ensure_wrapped(self):
        if self._wrapped:
            return
        wrap_linear_modules(
            self.model,
            self.target_modules,
            rank=self.lora_r,
            n_forget_experts=self.n_forget_experts,
            q=self.tsallis_q,
            high_entropy_cut=self.high_entropy_cut,
            tau_high=self.tau_high,
            tau_low=self.tau_low,
            top_k=self.top_k,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for module in iter_asym_modules(self.model):
            module.A.requires_grad = True
            module.B_r.requires_grad = True
            module.B_f.requires_grad = True
            for p in module.router.parameters():
                p.requires_grad = True
        self._wrapped = True

    def create_optimizer(self):
        self._ensure_wrapped()
        a_params, b_params = [], []
        for module in iter_asym_modules(self.model):
            a_params.append(module.A)
            b_params.extend([module.B_r, module.B_f, *module.router.parameters()])
        self.optimizer = torch.optim.Adam(
            [
                {"params": a_params, "lr": self.eta_a},
                {"params": b_params, "lr": self.eta_b},
            ]
        )
        return self.optimizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        self._ensure_wrapped()
        forget = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        retain = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }
        forget_out = model(**{k: v for k, v in forget.items() if k != "labels"})
        forget_loss = ihl_loss_from_logits(forget_out.logits, forget["labels"])

        retain_out = model(**retain)
        retain_loss = retain_out.loss

        entropy_reg = torch.zeros((), device=forget_loss.device)
        log_probs = F.log_softmax(forget_out.logits, dim=-1)
        probs = log_probs.exp()
        from open_unlearning_plugins.alter.asym_lora import tsallis_entropy

        token_s = tsallis_entropy(probs, self.tsallis_q)
        high = token_s > self.high_entropy_cut
        if high.any() and forget_out.logits.requires_grad:
            # λ ||∇_A S_q||^2 on high-entropy tokens — cheap proxy: variance of S_q.
            entropy_reg = token_s[high].pow(2).mean()

        loss = (
            self.beta * forget_loss
            + self.gamma_retain * retain_loss
            + self.lambda_entropy * entropy_reg
        )
        return (loss, forget_out) if return_outputs else loss
