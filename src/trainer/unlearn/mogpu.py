"""Frozen-candidate MOGP-U trainer."""

from __future__ import annotations

from pathlib import Path

from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import evaluate_loss, validate_candidate
from trainer.unlearn.mogpu_dsl.observables import deltas, model_inputs


class MOGPU(GradDiff):
    """Execute one statically-gated M-DSL candidate; evolution is external."""

    def __init__(self, candidate_spec_path, candidate_root=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_spec_path = Path(candidate_spec_path).resolve()
        root = (
            Path(candidate_root).resolve()
            if candidate_root
            else self.candidate_spec_path.parent
        )
        if (
            root not in self.candidate_spec_path.parents
            and root != self.candidate_spec_path.parent
        ):
            raise ValueError("candidate_spec_path must be inside candidate_root")
        self.candidate = validate_candidate(
            CandidateSpec.load(self.candidate_spec_path), root
        )
        # GradDiff only creates a reference for KL; MOGP-U observables always need one.
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = model_inputs(inputs["forget"])
        retain_inputs = model_inputs(inputs["retain"])
        delta_forget, delta_retain, forget_outputs = deltas(
            model, self.ref_model, forget_inputs, retain_inputs
        )
        loss = evaluate_loss(self.candidate, delta_forget, delta_retain)
        return (loss, forget_outputs) if return_outputs else loss
