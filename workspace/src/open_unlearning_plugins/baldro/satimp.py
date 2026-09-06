"""SatImp + BalDRO wrappers. Ported from nxZhai/BalDRO@5a93ede."""

from __future__ import annotations

from torch import nn

from trainer.unlearn.satimp import SatImp
from trainer.utils import compute_batch_nll, compute_satimp_loss

from open_unlearning_plugins.baldro.utils import (
    compute_forget_group,
    compute_retain_group,
    dv_aggregate,
)


class DrSatImp(SatImp):
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

    def compute_satimp_loss(self, model, inputs, beta1, beta2):
        outputs = model(**inputs)
        labels = inputs["labels"].to(outputs.logits.device)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        weight_sat = ((-lm_loss).exp().detach()) ** beta1
        weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
        weighted_token_loss = (weight_sat * weight_imp * lm_loss).view_as(shift_labels)
        loss_mask = shift_labels != -100
        valid_token_counts = loss_mask.sum(-1).clamp_min(1)
        forget_loss = -(weighted_token_loss * loss_mask).sum(-1) / valid_token_counts
        return forget_loss, outputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        forget_loss, forget_outputs = self.compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
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
            retain_mask = retain_inputs["labels"][..., 1:] != -100
            retain_loss = retain_loss / retain_mask.sum(-1).clamp_min(1)
            retain_loss = dv_aggregate(retain_loss, self.beta_dv_retain)
        else:
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupSatImp(SatImp):
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
        forget_loss, forget_outputs = compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
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
