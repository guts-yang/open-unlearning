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
        value = values.get(field["path"])
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or non-finite official metric: {field['path']}")
        result[name] = float(value) * (-1 if field.get("direction") == "min" else 1)
    if "forget_quality" not in values:
        raise ValueError("Official JSON lacks forget_quality")
    result["raw_forget_quality"] = float(values["forget_quality"])
    if not math.isfinite(result["raw_forget_quality"]):
        raise ValueError("forget_quality is non-finite")
    return result


def apply_objectives(
    record: CandidateRecord, values: dict[str, float], fq_threshold: float
) -> CandidateRecord:
    record.fq_feasible = values["raw_forget_quality"] >= fq_threshold
    record.constraint_violation = max(0.0, fq_threshold - values["raw_forget_quality"])
    record.objectives = {
        key: value for key, value in values.items() if key != "raw_forget_quality"
    }
    record.payload["forget_quality"] = values["raw_forget_quality"]
    return record
