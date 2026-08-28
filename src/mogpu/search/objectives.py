"""Official evaluator JSON parsing and constraint construction."""

import json
import math
from pathlib import Path

from mogpu.search.records import CandidateRecord


def parse_summary(
    path: str | Path, schema: dict, fq_threshold: float
) -> dict[str, float]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for name, field in schema.items():
        if field.get("synthetic") or field.get("optional") or field.get("constraint"):
            continue
        value = values.get(field["path"])
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or non-finite official metric: {field['path']}")
        result[name] = float(value) * (-1 if field.get("direction") == "min" else 1)
    if "forget_quality" not in values:
        raise ValueError("Official JSON lacks forget_quality")
    fq = values["forget_quality"]
    if fq is None or not isinstance(fq, (int, float)) or not math.isfinite(float(fq)):
        raise ValueError("forget_quality is non-finite")
    result["raw_forget_quality"] = float(fq)
    return result


def synthetic_complexity(record: CandidateRecord) -> float:
    atoms = record.canonical_spec.get("atoms") or []
    return -float(len(atoms))


def trajectory_stability(points: list[dict]) -> float:
    scores = [
        float(point["forget_score"])
        for point in points
        if isinstance(point.get("forget_score"), (int, float))
        and math.isfinite(point["forget_score"])
    ]
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return -math.sqrt(variance)


def apply_objectives(
    record: CandidateRecord, values: dict[str, float], fq_threshold: float
) -> CandidateRecord:
    record.fq_feasible = values["raw_forget_quality"] >= fq_threshold
    record.constraint_violation = max(0.0, fq_threshold - values["raw_forget_quality"])
    record.objectives = {
        key: value for key, value in values.items() if key != "raw_forget_quality"
    }
    record.objectives["negative_complexity"] = synthetic_complexity(record)
    points = record.payload.get("trajectory_summary", {}).get("points", [])
    record.objectives["negative_instability"] = trajectory_stability(points)
    record.payload["forget_quality"] = values["raw_forget_quality"]
    return record
