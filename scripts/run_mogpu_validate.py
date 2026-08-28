"""Validate a selected CandidateSpec for 10 epochs, then optional sealed forget05."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mogpu.search import SearchController
from mogpu.search.records import CandidateRecord
from mogpu.seed_registry import SeedRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/saves/mogpu_gpu_pilot"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _controller(output_dir: Path) -> SearchController:
    registry = SeedRegistry.load(ROOT / "configs/mogpu/seed_catalog.yaml")
    recipe = {
        "repo_root": str(ROOT),
        "output_dir": str(output_dir),
        "experiment": "unlearn/tofu/mogpu_search",
        "eval_experiment": "eval/tofu/default",
        "pretrained_model_name_or_path": (
            "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
        ),
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
    }
    return SearchController(registry, output_dir, recipe)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    summary_path = output_dir / "search_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; run search first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected")
    if selected is None:
        parents = summary.get("parents") or []
        if not parents:
            raise SystemExit("search produced no candidate to validate")
        selected = min(
            parents,
            key=lambda item: (
                float(item.get("constraint_violation", float("inf"))),
                item["candidate_hash"],
            ),
        )
        print(
            "No FQ-feasible pilot candidate; validating the least-violating "
            "candidate diagnostically."
        )
    record = CandidateRecord(
        candidate_hash=selected["candidate_hash"],
        canonical_spec=selected["canonical_spec"],
        generation=selected["generation"],
        parent_hashes=selected.get("parent_hashes") or [],
    )
    controller = _controller(output_dir)
    validated = controller.run_validation(record, seeds=[args.seed])
    (output_dir / "validation_summary.json").write_text(
        json.dumps([item.to_dict() for item in validated], indent=2) + "\n",
        encoding="utf-8",
    )
    sealed_logs = ROOT / "saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json"
    if sealed_logs.is_file():
        controller.recipe["retain_logs_path"] = str(sealed_logs)
        controller.evaluator.retain_logs_path = str(sealed_logs)
        sealed = controller.run_sealed_test(record, forget_split="forget05")
        (output_dir / "sealed_summary.json").write_text(
            json.dumps(sealed.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(f"skip sealed forget05; missing {sealed_logs}")
    (output_dir / "muse_summary.json").write_text(
        json.dumps(
            {
                "skipped": True,
                "reason": (
                    "MUSE stays on native metrics after a GPU-selected spec; "
                    "do not reuse the TOFU FQ gate."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
