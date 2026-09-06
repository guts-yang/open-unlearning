"""LoRA-BiAL-Adaptive trainer ported from tzpranto/blade@84f7504."""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from open_unlearning_plugins.blade.lora_bial import LoRABiAL

logger = logging.getLogger(__name__)


class LoRABiALAdaptive(LoRABiAL):
    def __init__(
        self,
        *args,
        calibration_frac: float = 0.10,
        conv_patience: int = 8,
        ema_alpha: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.calibration_frac = calibration_frac
        self.conv_patience = conv_patience
        self.ema_alpha = ema_alpha

    def _init_state(self):
        return {
            "lfgt_init": None,
            "lfgt_ema": None,
            "prev_lfgt_ema": None,
            "vel_ema": 0.0,
            "peak_vel": 0.0,
            "lr_adjusted": False,
            "lr_scale": 1.0,
            "r_breached_early": False,
            "calibration_step": None,
            "r_pos_streak": 0,
            "t_trans": None,
            "min_stop_step": None,
            "conv_count": 0,
            "stopped": False,
            "stop_step": None,
        }

    def _update_ema(self, state, l_fgt):
        alpha = self.ema_alpha
        if state["lfgt_init"] is None:
            state["lfgt_init"] = l_fgt
            state["lfgt_ema"] = l_fgt
            state["prev_lfgt_ema"] = l_fgt
            return

        state["lfgt_ema"] = alpha * l_fgt + (1 - alpha) * state["lfgt_ema"]
        vel = state["lfgt_ema"] - state["prev_lfgt_ema"]
        state["vel_ema"] = alpha * vel + (1 - alpha) * state["vel_ema"]
        state["prev_lfgt_ema"] = state["lfgt_ema"]
        if abs(state["vel_ema"]) > state["peak_vel"]:
            state["peak_vel"] = abs(state["vel_ema"])

    def _calibrate_lr(self, step, l_fgt, residual, state, max_steps):
        del l_fgt
        cal_step = max(5, int(self.calibration_frac * max_steps))
        state["calibration_step"] = cal_step

        if state["lr_adjusted"] or step < cal_step:
            if residual > self.epsilon and step < cal_step:
                state["r_breached_early"] = True
            return

        state["lr_adjusted"] = True
        frac_remaining = state["lfgt_ema"] / state["lfgt_init"]
        observed_rate = -math.log(max(frac_remaining, 0.01)) / step
        target_rate = -math.log(0.5) / (max_steps / 3.0)

        if state["r_breached_early"]:
            scale = 0.5
            reason = f"r breached ε={self.epsilon:.4f} during calibration"
        else:
            scale = target_rate / max(observed_rate, 1e-8)
            scale = max(0.3, min(3.0, scale))
            reason = (
                f"obs_rate={observed_rate:.5f}, target={target_rate:.5f}, "
                f"L_fgt dropped to {frac_remaining:.1%}"
            )

        if abs(scale - 1.0) > 0.08:
            old_lr = self._outer_opt.param_groups[0]["lr"]
            new_lr = old_lr * scale
            for pg in self._outer_opt.param_groups:
                pg["lr"] = new_lr
            state["lr_scale"] = scale
            logger.info(
                f" LR-CALIBRATE @ step {step}: {old_lr:.2e} → {new_lr:.2e} "
                f"(×{scale:.2f}, {reason})"
            )
        else:
            logger.info(f" LR-CALIBRATE @ step {step}: no adjustment ({reason})")

    def _detect_transition(self, step, residual, state):
        if state["t_trans"] is not None:
            return
        if residual > self.epsilon:
            state["r_pos_streak"] += 1
            if state["r_pos_streak"] >= 3:
                state["t_trans"] = step - 2
                state["min_stop_step"] = int(state["t_trans"] * 1.6)
                logger.info(
                    f" PHASE-TRANS @ step {step}: T_trans={state['t_trans']}, "
                    f"min_stop={state['min_stop_step']}"
                )
        else:
            state["r_pos_streak"] = 0

    def _check_convergence(self, step, state):
        if state["t_trans"] is None:
            return False
        if state["min_stop_step"] is not None and step < state["min_stop_step"]:
            return False
        if state["peak_vel"] < 1e-8:
            return False

        drop_frac = 1.0 - (state["lfgt_ema"] / state["lfgt_init"])
        if drop_frac < 0.50:
            return False

        vel_ratio = abs(state["vel_ema"]) / state["peak_vel"]
        lfgt_ratio = state["lfgt_ema"] / state["lfgt_init"]
        if vel_ratio < 0.01 or lfgt_ratio < 0.02:
            state["conv_count"] += 1
            if state["conv_count"] >= self.conv_patience:
                state["stopped"] = True
                state["stop_step"] = step - self.conv_patience + 1
                return True
        else:
            state["conv_count"] = 0
        return False

    def train(self, *args, **kwargs):
        del args, kwargs
        device = self.args.device

        logger.info("=" * 60)
        logger.info("LoRA-BiAL Adaptive (self-tuning LR + convergence)")
        logger.info("=" * 60)
        self._wrap_with_lora()

        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)

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

        forget_micro_batches = len(self._forget_dataloader)
        steps_per_epoch = max(
            1, forget_micro_batches // self.gradient_accumulation_steps
        )
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_outer_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        history = []
        log_every = max(1, max_outer_steps // 30)
        state = self._init_state()
        resume_step = 0

        if self.checkpoint_every_n_steps > 0:
            ckpt_dirs = sorted(
                glob.glob(os.path.join(self.args.output_dir, "checkpoint-step*"))
            )
            if ckpt_dirs:
                latest_ckpt = ckpt_dirs[-1]
                resume_step = int(latest_ckpt.split("checkpoint-step")[-1])
                from peft import PeftModel

                if isinstance(self.model, PeftModel):
                    self.model.load_adapter(latest_ckpt, adapter_name="default")
                else:
                    self.model = PeftModel.from_pretrained(self.model, latest_ckpt)
                self.model.train()

        inner_losses = self.inner_loop(device)
        if inner_losses:
            baseline_ret = sum(inner_losses) / len(inner_losses)
        else:
            with torch.no_grad():
                batch = self._next_retain_batch()
                baseline_ret = self._compute_ce_loss(batch, device).item()
        self.epsilon = self.epsilon_multiplier * baseline_ret
        logger.info(
            f" Auto-ε: inner_avg={baseline_ret:.4f}, "
            f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}"
        )

        for t in range(resume_step, max_outer_steps):
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

            dt = time.time() - t_start
            inner_mean = (
                sum(inner_losses) / len(inner_losses) if inner_losses else 0.0
            )

            self._update_ema(state, l_fgt)
            self._calibrate_lr(t, l_fgt, residual, state, max_outer_steps)
            self._detect_transition(t, residual, state)
            converged = self._check_convergence(t, state)

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
                    "lr": self._outer_opt.param_groups[0]["lr"],
                    "lfgt_ema": state["lfgt_ema"]
                    if state["lfgt_ema"] is not None
                    else l_fgt,
                    "vel_ema": state["vel_ema"],
                    "peak_vel": state["peak_vel"],
                    "t_trans": state["t_trans"],
                    "converged": state["stopped"],
                }
            )

            if t % log_every == 0 or t == max_outer_steps - 1 or converged:
                logger.info(
                    f" [{t:4d}/{max_outer_steps}|e{epoch + 1}] "
                    f"L_fgt={l_fgt:.4f} L_ret={l_ret:.4f} "
                    f"r={residual:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={inner_mean:.4f} dt={dt:.1f}s"
                )

            if converged:
                ckpt_dir = os.path.join(self.args.output_dir, "converged-best")
                self._save_checkpoint(ckpt_dir, history)
                break

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
            if l_ret > 10.0:
                logger.warning(
                    f" L_ret={l_ret:.1f} > 10.0 — collapsed at step {global_step}."
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
