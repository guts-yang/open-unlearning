"""GPU smoke: one enabled seed through F0–F2 under the frozen 10-step recipe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mogpu.search import SearchController
from mogpu.seed_registry import SeedRegistry


def main() -> None:
    output_dir = Path("/root/autodl-tmp/saves/mogpu")
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    recipe = {
        "repo_root": str(ROOT),
        "output_dir": str(output_dir),
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "protocol_dir": str(ROOT / "configs/mogpu/search"),
        "retain_logs_path": str(
            ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
        ),
        "fq_threshold": 0.05,
        "training_seed": 0,
        "lora_rank": None,
        "searchable": [],
        "dry_run": False,
        "budget": {"max_steps": 10},
        "forget_split": "forget10",
        "cache_policy": "reuse_succeeded_only",
        "nsga_overrides": {
            "population_size": 1,
            "offspring_size": 1,
            "max_generations": 1,
            "archive_capacity": 4,
        },
    }
    controller = SearchController(registry, output_dir, recipe)
    record = controller.run_seed("SIMNPO_ER_01")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "hash": record.candidate_hash,
                "fq": record.payload.get("forget_quality"),
                "feasible": record.fq_feasible,
            }
        )
    )


if __name__ == "__main__":
    main()
