"""LoRA-BiAL trainer ported from tzpranto/blade@84f7504.

Adapted for this OpenUnlearning checkout: HuggingFace Trainer uses
``processing_class`` instead of ``tokenizer``. Official src/ is not modified.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

from trainer.unlearn.base import UnlearnTrainer

from open_unlearning_plugins.blade.lora_bial_losses import (
    FORGET_LOSS_DISPATCH,
    compute_ce_loss,
)

logger = logging.getLogger(__name__)


class LoRABiAL(UnlearnTrainer):
    def __init__(
        self,
        *args,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        T: int = -1,
        K: int = 3,
        eta_theta: float = 3e-5,
        eta_in: float = 2e-4,
        epsilon_multiplier: float = 1.15,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 0.0,
        lambda_min: float = 0.1,
        dual_decay_factor: float = 0.1,
        forget_loss_type: str = "clamped_entropy",
        npo_beta: float = 4.0,
        clamped_entropy_tau: float = 0.7,
        checkpoint_every_epoch: bool = False,
        checkpoint_every_n_steps: int = 0,
        eval_at_steps: Optional[list] = None,
        lora_init_path: Optional[str] = None,
        save_lora_only: bool = False,
        gradient_accumulation_steps: int = 8,
        inner_accumulation_steps: int = 0,
        inner_warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        self.epsilon = None
        self.epsilon_multiplier = epsilon_multiplier
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.dual_decay_factor = dual_decay_factor
        self.lambda_dual = float(lambda_init)
        self.forget_loss_type = forget_loss_type
        if forget_loss_type not in FORGET_LOSS_DISPATCH:
            raise ValueError(
                f"Unknown forget_loss_type: {forget_loss_type}. "
                f"Available: {list(FORGET_LOSS_DISPATCH.keys())}"
            )
        self.npo_beta = npo_beta
        self.clamped_entropy_tau = clamped_entropy_tau
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()
        self.lora_init_path = lora_init_path
        self.save_lora_only = save_lora_only
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.inner_accumulation_steps = (
            inner_accumulation_steps
            if inner_accumulation_steps > 0
            else gradient_accumulation_steps
        )
        self.inner_warmup_steps = inner_warmup_steps
        self.max_grad_norm = max_grad_norm

    def _get_tokenizer(self):
        return getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)

    def _wrap_with_lora(self):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if self.lora_init_path and os.path.isdir(self.lora_init_path):
            logger.info(f"Loading LoRA adapters from: {self.lora_init_path}")
            self.model = PeftModel.from_pretrained(
                self.model, self.lora_init_path, is_trainable=True
            )
        else:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha_val,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        logger.info(
            f"LoRA applied: r={self.lora_r}, alpha={self.lora_alpha_val}, "
            f"modules={self.lora_target_modules}"
        )

    def _compute_ce_loss(self, batch, device):
        return compute_ce_loss(self.model, batch, device)

    def _compute_forget_loss(self, batch, device):
        return FORGET_LOSS_DISPATCH[self.forget_loss_type](
            self.model,
            batch,
            device,
            npo_beta=self.npo_beta,
            clamped_entropy_tau=self.clamped_entropy_tau,
        )

    def _next_retain_batch(self):
        try:
            return next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self._retain_dataloader)
            return next(self._retain_iter)

    def _next_forget_batch(self):
        try:
            return next(self._forget_iter)
        except StopIteration:
            self._forget_iter = iter(self._forget_dataloader)
            return next(self._forget_iter)

    def inner_loop(self, device):
        self.model.enable_adapter_layers()
        self.model.train()
        inner_losses = []
        for _k in range(self.K):
            self._inner_opt.zero_grad()
            accum_loss = 0.0
            for _ in range(self.inner_accumulation_steps):
                batch = self._next_retain_batch()
                loss = self._compute_ce_loss(batch, device)
                (loss / self.inner_accumulation_steps).backward()
                accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._inner_opt.step()
            inner_losses.append(accum_loss / self.inner_accumulation_steps)
        return inner_losses

    def outer_step(self, device, global_step=0):
        del global_step
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        total_l_fgt = 0.0
        total_l_ret = 0.0

        for _ in range(self.gradient_accumulation_steps):
            forget_batch = self._next_forget_batch()
            retain_batch = self._next_retain_batch()

            l_fgt = self._compute_forget_loss(forget_batch, device)
            l_ret = self._compute_ce_loss(retain_batch, device)

            r_micro = l_ret - self.epsilon
            r_plus = torch.clamp(r_micro, min=0.0)
            l_alm = l_fgt + self.lambda_dual * r_micro + 0.5 * self.rho * (r_plus**2)
            (l_alm / self.gradient_accumulation_steps).backward()

            total_l_fgt += l_fgt.item()
            total_l_ret += l_ret.item()

        avg_l_fgt = total_l_fgt / self.gradient_accumulation_steps
        avg_l_ret = total_l_ret / self.gradient_accumulation_steps
        avg_r = avg_l_ret - self.epsilon

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self._outer_opt.step()

        if avg_r > 0:
            self.lambda_dual += self.rho * avg_r
        else:
            self.lambda_dual += self.dual_decay_factor * self.rho * avg_r
        self.lambda_dual = max(self.lambda_min, self.lambda_dual)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return avg_l_fgt, avg_l_ret, avg_r

    def _save_checkpoint(self, ckpt_dir, history):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.eval()
        self.model.merge_adapter()
        peft_sd = self.model.base_model.model.state_dict()
        clean_sd = {
            k.replace(".base_layer", ""): v
            for k, v in peft_sd.items()
            if "lora_" not in k.replace(".base_layer", "")
        }
        self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_dir)
        self.model.unmerge_adapter()
        self.model.train()
        with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as handle:
            json.dump(history, handle, indent=2)

    def train(self, *args, **kwargs):
        del args, kwargs
        device = self.args.device

        logger.info("=" * 60)
        logger.info("LoRA adapters")
        logger.info("=" * 60)
        self._wrap_with_lora()

        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        logger.info(
            f"Optimizers: inner=SGD(lr={self.eta_in}), outer=Adam(lr={self.eta_theta}), "
            f"params={sum(p.numel() for p in lora_params)}"
        )

        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        forget_ds = self.train_dataset.forget
        retain_ds = self.train_dataset.retain
        collator = self.data_collator
        self._forget_dataloader = DataLoader(
            forget_ds,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )
        self._retain_dataloader = DataLoader(
            retain_ds,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )
        self._forget_iter = iter(self._forget_dataloader)
        self._retain_iter = iter(self._retain_dataloader)

        batch_size = self.args.per_device_train_batch_size
        forget_micro_batches = len(self._forget_dataloader)
        steps_per_epoch = max(1, forget_micro_batches // self.gradient_accumulation_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_outer_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        logger.info(
            f" Mode: {'epoch-based' if self.T <= 0 else f'fixed T={self.T}'}"
        )
        logger.info(
            f" Forget loss: {self.forget_loss_type}  ALM ε_mul={self.epsilon_multiplier}"
        )

        history = []
        log_every = max(1, max_outer_steps // 20)

        inner_losses = self.inner_loop(device)
        baseline_ret = sum(inner_losses) / len(inner_losses)
        self.epsilon = self.epsilon_multiplier * baseline_ret
        logger.info(
            f" Auto-ε: inner_avg={baseline_ret:.4f}, "
            f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}"
        )

        for t in range(max_outer_steps):
            t_start = time.time()
            epoch = t // steps_per_epoch if steps_per_epoch > 0 else 0

            if t == 0:
                inner_losses = []
            elif t < self.inner_warmup_steps:
                inner_losses = []
            else:
                inner_losses = self.inner_loop(device)

            l_fgt, l_ret, residual = self.outer_step(device, global_step=t)

            extra_inner = 0
            while l_ret > 2 * self.epsilon and extra_inner < self.K * 3:
                self.inner_loop(device)
                with torch.no_grad():
                    retain_batch = self._next_retain_batch()
                    l_ret = self._compute_ce_loss(retain_batch, device).item()
                    residual = l_ret - self.epsilon
                extra_inner += self.K
            if extra_inner > 0:
                logger.info(
                    f" Adaptive inner: {extra_inner} extra steps, L_ret={l_ret:.4f}"
                )

            dt = time.time() - t_start
            inner_mean = (
                sum(inner_losses) / len(inner_losses) if inner_losses else 0.0
            )
            history.append(
                {
                    "step": t,
                    "epoch": epoch,
                    "L_fgt": l_fgt,
                    "L_ret": l_ret,
                    "r": residual,
                    "lambda": self.lambda_dual,
                    "inner_loss_mean": inner_mean,
                    "dt": dt,
                }
            )

            if t % log_every == 0 or t == max_outer_steps - 1:
                logger.info(
                    f" [{t:4d}/{max_outer_steps}|e{epoch + 1}] "
                    f"L_fgt={l_fgt:.4f} L_ret={l_ret:.4f} "
                    f"r={residual:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={inner_mean:.4f} dt={dt:.1f}s"
                )

            global_step = t + 1
            if global_step in self.eval_at_steps:
                ckpt_dir = os.path.join(self.args.output_dir, f"step-{global_step}")
                self._save_checkpoint(ckpt_dir, history)
            if (
                self.checkpoint_every_n_steps > 0
                and global_step % self.checkpoint_every_n_steps == 0
                and global_step < max_outer_steps
            ):
                ckpt_dir = os.path.join(
                    self.args.output_dir, f"checkpoint-step{global_step}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
            if (
                self.checkpoint_every_epoch
                and steps_per_epoch > 0
                and global_step % steps_per_epoch == 0
                and global_step < max_outer_steps
            ):
                ep_num = global_step // steps_per_epoch
                ckpt_dir = os.path.join(
                    self.args.output_dir, f"checkpoint-epoch{ep_num}"
                )
                self._save_checkpoint(ckpt_dir, history)

            if l_ret > 10.0:
                logger.warning(
                    f" L_ret={l_ret:.1f} > 10.0 — model collapsed at step {global_step}."
                )
                break

        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.save_lora_only:
            lora_dir = os.path.join(output_dir, "lora_adapters")
            os.makedirs(lora_dir, exist_ok=True)
            self.model.save_pretrained(lora_dir)

        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(output_dir)
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "lora_bial_history.json"), "w") as handle:
            json.dump(history, handle, indent=2)
        self.evaluate()
        return None
