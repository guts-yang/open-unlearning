import torch
import torch.nn.functional as F
from torch import nn

from trainer.unlearn.grad_diff import GradDiff


class TPO(GradDiff):
    """Targeted Preference Optimization (TPO) trainer.

    From "Not All Tokens Are Meant to Be Forgotten" (AAAI 2026, arXiv:2506.03142).
    Official implementation: https://github.com/guts-yang/Unlearning-TPO

    TPO splits the forget-set answer tokens into Unwanted Words (UW, targeted for
    forgetting) and General Words (GW, to be preserved) via offline annotations
    (`common_words` spans, provided by the `QAwithCommonWordsDataset` data handler).
    The loss combines:

    - a logit preference loss on the UW tokens: an NPO-style sigmoid term over the
      difference between the average oracle (reference model) raw logits and the
      current model raw logits, where the oracle's logits on the true next tokens
      are masked to -1e4 (port of the official `loss_type == 'tpo'` branch);
    - a preservation loss: NLL on the GW tokens, keeping general language ability
      intact while the targeted knowledge is forgotten;
    - optionally (``alpha > 0``), a retain-set NLL term (the official
      ``tpo_grad_diff`` variant). Default ``alpha=0`` reproduces plain TPO.

    Note: `gamma` is inherited from GradDiff but unused, as in the official loss.
    """

    def __init__(self, beta=0.3, alpha=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.alpha = alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    @staticmethod
    def _get_batch_loss(logits, labels):
        """Per-sequence sum of token-level cross entropy over active label positions.

        Mirrors `get_batch_loss` in the official implementation
        (TOFU/data_module.py)."""
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        return loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        input_ids = forget_inputs["input_ids"]
        attention_mask = forget_inputs["attention_mask"]
        labels = forget_inputs["labels"]  # active only on UW tokens
        gw_labels = forget_inputs["gw_labels"]  # active only on GW tokens

        # preservation loss: NLL on the general words of the forget answers
        gw_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=gw_labels
        )
        pl_loss = self._get_batch_loss(gw_outputs.logits, gw_labels).mean()

        # current policy logits on the unwanted words
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # oracle (reference fine-tuned model) logits, with the logits of the true
        # next tokens masked out (official implementation sets them to -1e4)
        with torch.no_grad():
            forget_outputs_oracle = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            forget_logits_oracle = forget_outputs_oracle.logits.clone()
            batch_size, seq_len = input_ids[:, 1:].size()
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            seq_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            forget_logits_oracle[batch_idx, seq_idx, input_ids[:, 1:]] = float(-1e4)

        input_ids_expanded = input_ids[:, 1:].unsqueeze(-1)
        logits = outputs.logits[:, :-1, :]
        logits_oracle = forget_logits_oracle[:, :-1, :]
        loss_indexes = (labels[:, 1:] != -100).float()
        # normalize per sequence (average logit over UW positions); clamp guards
        # against sequences without any UW token
        loss_indexes = loss_indexes / loss_indexes.sum(-1, keepdim=True).clamp(min=1.0)
        lpl = (
            (
                torch.gather(logits_oracle, dim=-1, index=input_ids_expanded).squeeze(-1)
                * loss_indexes
            ).sum(-1)
            - (
                torch.gather(logits, dim=-1, index=input_ids_expanded).squeeze(-1)
                * loss_indexes
            ).sum(-1)
        ).mean()

        loss = -F.logsigmoid(self.beta * lpl) * 2 / self.beta + pl_loss

        # optional retain-set term (official `tpo_grad_diff` variant)
        if self.alpha > 0:
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )
            loss = loss + self.alpha * retain_loss

        return (loss, outputs) if return_outputs else loss
