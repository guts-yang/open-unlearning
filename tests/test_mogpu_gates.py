import math

import pytest

from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def valid_spec(**overrides):
    data = {
        "atoms": ["EraseResidual", "RetainDrift"],
        "weights": [0.5, 0.5],
        "thresholds": {"kappa": 0.3, "tau": 0.05},
        "approved_variants": {"EraseResidual": "softplus", "RetainDrift": "huber"},
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "data",
    [
        valid_spec(weights=[-1.0, 2.0]),
        valid_spec(weights=[math.nan, 1.0]),
        valid_spec(atoms=["EraseResidual"]),
        valid_spec(
            atoms=["EraseResidual", "RetainDrift", "SelectiveMargin", "EraseResidual"]
        ),
        valid_spec(thresholds={"temperature": 0.0}),
    ],
)
def test_invalid_candidates_are_rejected(data):
    with pytest.raises(ValueError):
        validate_candidate(CandidateSpec.from_dict(data))


def test_unknown_expression_field_is_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        CandidateSpec.from_dict(valid_spec(expression="logits / x"))
