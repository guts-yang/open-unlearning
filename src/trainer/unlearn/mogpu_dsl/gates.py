"""Static and numerical safety gates for closed MOGP-U candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from trainer.unlearn.mogpu_dsl import atoms
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec

FORBIDDEN_CONFIG_KEYS = ("lora_rank", "peft", "adapter", "learning_rate", "batch_size")


def validate_candidate(
    spec: CandidateSpec, allowed_root: str | Path | None = None
) -> CandidateSpec:
    spec = spec.canonical()
    thresholds = dict(spec.thresholds)
    for key in ("temperature", "tau", "epsilon"):
        if thresholds.get(key, 1.0) <= 0:
            raise ValueError(f"{key} must be positive")
    if thresholds.get("kappa", 0.0) < 0:
        raise ValueError("kappa must be non-negative")
    if allowed_root is not None and not Path(allowed_root).is_dir():
        raise ValueError("Candidate root must exist before loading candidates")
    _verify_gradients(spec)
    return spec


def validate_hydra_config(config: Any) -> None:
    """Reject recipe genes accidentally smuggled into a candidate-facing config."""

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_path = f"{path}.{key}" if path else str(key)
                if any(
                    token in str(key).lower() for token in ("peft", "adapter", "lora")
                ):
                    if key_path.endswith("lora_rank") and nested is None:
                        continue
                    raise ValueError(
                        f"MOGP-U config forbids adapter override: {key_path}"
                    )
                visit(nested, key_path)

    visit(config)


def _loss(spec: CandidateSpec, delta_forget: torch.Tensor, delta_retain: torch.Tensor):
    values = []
    for term in spec.terms:
        if term.atom == "EraseResidual":
            value = atoms.erase_residual(
                delta_forget,
                spec.threshold("kappa", 0.0),
                spec.threshold("temperature", 1.0),
            )
        elif term.atom == "RetainDrift":
            value = atoms.retain_drift(delta_retain, spec.threshold("tau", 0.05))
        else:
            value = atoms.selective_margin(
                delta_forget,
                delta_retain,
                spec.threshold("kappa", 0.0),
                spec.threshold("temperature", 1.0),
                spec.threshold("epsilon", 1e-6),
            )
        values.append(term.weight * value)
    loss = torch.stack(values).sum(dim=0).mean()
    if not torch.isfinite(loss):
        raise ValueError("MOGP-U candidate produced a non-finite loss")
    return loss


def evaluate_loss(
    spec: CandidateSpec, delta_forget: torch.Tensor, delta_retain: torch.Tensor
):
    return _loss(spec, delta_forget, delta_retain)


def _verify_gradients(spec: CandidateSpec) -> None:
    for retain_value in (-1.0, -0.1, 0.0, 0.1, 1.0):
        forget = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64, requires_grad=True)
        retain = torch.full_like(forget, retain_value, requires_grad=True)
        loss = _loss(spec, forget, retain)
        grad_forget, grad_retain = torch.autograd.grad(loss, (forget, retain))
        if (
            not torch.isfinite(grad_forget).all()
            or not torch.isfinite(grad_retain).all()
        ):
            raise ValueError("MOGP-U candidate has non-finite gradients")
        if torch.any(grad_forget < -1e-10):
            raise ValueError("MOGP-U candidate violates forget gradient direction")
        if torch.any(retain * grad_retain < -1e-10):
            raise ValueError("MOGP-U candidate violates retain gradient direction")
