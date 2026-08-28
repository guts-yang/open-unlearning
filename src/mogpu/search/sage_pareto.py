"""Evidence-gated F0/F1/F2/F3 promotion state machine."""

from __future__ import annotations

import math

from mogpu.search.records import CandidateRecord
from trainer.unlearn.mogpu_dsl.gates import validate_candidate


def spec_from_record(record: CandidateRecord):
    from trainer.unlearn.mogpu_dsl.ast import CandidateSpec

    return CandidateSpec.from_dict(record.canonical_spec)


def f0(record: CandidateRecord) -> CandidateRecord:
    try:
        validate_candidate(spec_from_record(record))
        record.stage, record.status, record.constraint_violation = "F0", "passed", 0.0
    except (KeyError, ValueError, TypeError) as error:
        record.stage, record.status, record.constraint_violation = (
            "F0",
            "static_invalid",
            math.inf,
        )
        record.payload["failure_reason"] = str(error)
    return record


def action_evidence(
    record: CandidateRecord, evidence: dict[str, float]
) -> CandidateRecord:
    required = ("local_forget_action", "local_retain_action", "gradient_norm")
    valid = all(math.isfinite(float(evidence.get(key, math.nan))) for key in required)
    valid = (
        valid
        and evidence["local_forget_action"] >= 0
        and evidence["local_retain_action"] >= 0
        and evidence["gradient_norm"] > 0
    )
    record.stage, record.status = "F1", "passed" if valid else "action_invalid"
    record.payload["action_evidence"] = evidence
    return record


def trajectory(record: CandidateRecord, points: list[dict]) -> CandidateRecord:
    record.stage, record.status = "F2", "passed" if points else "runtime_invalid"
    record.payload["trajectory_summary"] = {
        "points": points,
        "kind": "trajectory_stability",
    }
    return record


def promote_f3(
    record: CandidateRecord, sample_count: int, minimum_samples: int
) -> bool:
    return (
        record.status == "passed"
        and record.fq_feasible
        and sample_count >= minimum_samples
    )
