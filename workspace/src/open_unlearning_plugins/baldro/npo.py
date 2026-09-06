"""NPO + BalDRO wrappers. Ported from nxZhai/BalDRO@5a93ede, adapted to this fork."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import NPO
from trainer.utils import compute_batch_nll, compute_dpo_loss

from open_unlearning_plugins.baldro.utils import (
    compute_forget_group,
    compute_retain_group,
    dv_aggregate,
)


class DrNPO(NPO):
    def __init__(
        self,
        beta_dv_forget=1.0,
        beta_dv_retain=1.0,
        forget_dro=True,
        retain_dro=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta_dv_forget = beta_dv_forget
        self.beta_dv_retain = beta_dv_retain
        self.forget_dro = forget_dro
        self.retain_dro = retain_dro

    def compute_dpo_loss(
        self, model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0
    ):
        if win_inputs is None and lose_inputs is None:
            raise ValueError("Both win_inputs and lose_inputs can't be None")

        win_log_ratio, lose_log_ratio = 0.0, 0.0
        win_outputs, lose_outputs = None, None

        if win_inputs is not None:
            win_loss, win_outputs = compute_batch_nll(model, win_inputs)
            with torch.no_grad():
                win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
            win_log_ratio = -(win_loss - win_ref_loss)

        if lose_inputs is not None:
            lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
            with torch.no_grad():
                lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
            lose_log_ratio = -(lose_loss - lose_ref_loss)

        loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio))
        return loss, (win_outputs, lose_outputs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = inputs["forget"]
        forget_loss, forget_outputs = self.compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        if self.forget_dro:
            forget_loss = dv_aggregate(forget_loss, self.beta_dv_forget)
        else:
            forget_loss = forget_loss.mean()

        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }
        if self.retain_dro:
            if self.retain_loss_type != "NLL":
                raise AssertionError("DRO only supports NLL retain loss currently.")
            retain_loss, _ = compute_batch_nll(model, retain_inputs)
            shifted_labels = retain_inputs["labels"][..., 1:]
            valid_token_counts = shifted_labels.ne(-100).sum(dim=-1)
            retain_loss = retain_loss / valid_token_counts.clamp_min(1)
            retain_loss = dv_aggregate(retain_loss, self.beta_dv_retain)
        else:
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupNPO(NPO):
    def __init__(self, sampling_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_ratio = sampling_ratio

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        forget_inputs, k, _top_idx = compute_forget_group(
            model, forget_inputs, self.sampling_ratio
        )
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        retain_inputs = compute_retain_group(
            {
                "input_ids": inputs["retain"]["input_ids"],
                "attention_mask": inputs["retain"]["attention_mask"],
                "labels": inputs["retain"]["labels"],
            },
            k,
        )
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
