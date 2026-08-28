"""Human-readable explanation cards for immutable MOGP-U candidates."""

from trainer.unlearn.mogpu_dsl.ast import CandidateSpec

_EXPLANATIONS = {
    "EraseResidual": "penalizes forget-answer residual likelihood and lowers Delta_f",
    "RetainDrift": "penalizes retain deviation and restores Delta_r toward zero",
    "SelectiveMargin": "requires forget change to exceed retain drift by a margin",
}


def render_card(spec: CandidateSpec, source_method: str | None = None) -> dict:
    return {
        "ast_hash": spec.ast_hash,
        "source_method": source_method,
        "terms": [
            {
                "atom": term.atom,
                "weight": term.weight,
                "variant": term.variant,
                "explanation": _EXPLANATIONS[term.atom],
            }
            for term in spec.terms
        ],
        "thresholds": dict(spec.thresholds),
        "safety_contract": {
            "forget": "dL/dDelta_f >= 0",
            "retain": "Delta_r * dL/dDelta_r >= 0",
        },
    }
