"""SimNPO + BalDRO wrappers. Ported from nxZhai/BalDRO@5a93ede."""

from __future__ import annotations

import torch.nn.functional as F

from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll

from open_unlearning_plugins.baldro.utils import (
    compute_forget_group,
    compute_retain_group,
    dv_aggregate,
)


class DrSimNPO(SimNPO):
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = inputs["forget"]
        loss_mask = forget_inputs["labels"][..., 1:] != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1).clamp_min(1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss) * 2 / self.beta
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
            retain_mask = retain_inputs["labels"][..., 1:] != -100
            retain_loss = retain_loss / retain_mask.sum(-1).clamp_min(1)
            retain_loss = dv_aggregate(retain_loss, self.beta_dv_retain)
        else:
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupSimNPO(SimNPO):
    def __init__(self, sampling_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_ratio = sampling_ratio

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs, k, _top_idx = compute_forget_group(
            model, inputs["forget"], self.sampling_ratio
        )
        forget_labels = forget_inputs["labels"][..., 1:].contiguous()
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
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
