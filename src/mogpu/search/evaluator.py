"""Train/eval a frozen CandidateSpec under the official TOFU protocol."""

from __future__ import annotations

import json
import math
import socket
import subprocess
from hashlib import sha256
from pathlib import Path

from mogpu.ledger import Ledger
from mogpu.search.objectives import apply_objectives, parse_summary
from mogpu.search.records import CandidateRecord
from mogpu.search.sage_pareto import trajectory
from trainer.unlearn.mogpu_dsl.ast import CandidateSpec
from trainer.unlearn.mogpu_dsl.render import render_card


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FitnessEvaluator:
    def __init__(self, recipe: dict, schema: dict, fq_threshold: float):
        self.recipe = recipe
        self.schema = schema
        self.fq_threshold = fq_threshold
        self.repo_root = Path(recipe["repo_root"])
        self.output_dir = Path(recipe["output_dir"])
        self.retain_logs_path = recipe["retain_logs_path"]
        self.dry_run = bool(recipe.get("dry_run", False))

    def cache_key(self, record: CandidateRecord, max_steps: int, seed: int) -> tuple:
        return (record.candidate_hash, max_steps, seed, record.tier)

    def __call__(self, record: CandidateRecord) -> CandidateRecord:
        budget = self.recipe.get("stage_budget") or self.recipe["budget"]
        max_steps = int(budget.get("max_steps", 10))
        seed = int(self.recipe.get("training_seed", 0))
        return self.evaluate(record, max_steps=max_steps, seed=seed)

    def evaluate(
        self, record: CandidateRecord, max_steps: int, seed: int
    ) -> CandidateRecord:
        if self.dry_run:
            return self._dry_run(record, max_steps, seed)
        spec = CandidateSpec.from_dict(record.canonical_spec)
        trial = (
            self.output_dir
            / record.tier
            / f"g{record.generation}_{spec.ast_hash[:12]}_s{seed}_n{max_steps}"
        )
        spec_path = self.output_dir / "candidates" / f"{spec.ast_hash}.json"
        write_spec(spec, spec_path)
        trial.mkdir(parents=True, exist_ok=True)
        final_eval = trial / f"evals-{max(max_steps, 0)}" / "TOFU_SUMMARY.json"
        trained_model = trial / "config.json"
        if not (
            self.recipe.get("cache_policy") == "reuse_succeeded_only"
            and (final_eval.is_file() or trained_model.is_file())
        ):
            self._train(spec_path, trial, max_steps, seed)
        points = []
        seen: set[str] = set()
        targets: list[tuple[int, Path]] = []
        if max_steps > 0:
            for step in self.recipe.get("checkpoint_steps", [max_steps]):
                if int(step) < max_steps:
                    ckpt = trial / f"checkpoint-{int(step)}"
                    if ckpt.exists():
                        targets.append((int(step), ckpt))
        targets.append((max(max_steps, 0), trial))
        for step, ckpt in targets:
            marker = str(ckpt)
            if marker in seen:
                continue
            seen.add(marker)
            eval_dir = trial / f"evals-{step}"
            summary = eval_dir / "TOFU_SUMMARY.json"
            if not (
                self.recipe.get("cache_policy") == "reuse_succeeded_only"
                and summary.is_file()
            ):
                summary = self._eval(ckpt, eval_dir)
            parsed = parse_summary(summary, self.schema, self.fq_threshold)
            points.append(
                {
                    "step": int(step),
                    "forget_score": parsed.get("forget_score"),
                    "retain_utility": parsed.get("retain_utility"),
                    "forget_quality": parsed["raw_forget_quality"],
                    "summary": str(summary),
                }
            )
        if not points:
            raise RuntimeError(f"No TOFU summaries produced for {trial}")
        record = trajectory(record, points)
        record = apply_objectives(
            record,
            {
                "forget_score": points[-1]["forget_score"],
                "retain_utility": points[-1]["retain_utility"],
                "raw_forget_quality": points[-1]["forget_quality"],
            },
            self.fq_threshold,
        )
        record.payload["output_dir"] = str(trial)
        record.payload["evaluator_json_path"] = points[-1]["summary"]
        self._append_ledger(record, spec, trial, points[-1]["summary"])
        return record

    def _dry_run(
        self, record: CandidateRecord, max_steps: int, seed: int
    ) -> CandidateRecord:
        digest = sha256(f"{record.candidate_hash}:{max_steps}:{seed}".encode()).digest()
        fq = digest[0] / 255.0
        utility = 0.4 + digest[1] / 2550.0
        score = 0.3 + digest[2] / 850.0
        points = [
            {
                "step": max(max_steps, 0),
                "forget_score": score,
                "retain_utility": utility,
                "forget_quality": fq,
            }
        ]
        record = trajectory(record, points)
        record = apply_objectives(
            record,
            {
                "forget_score": score,
                "retain_utility": utility,
                "raw_forget_quality": fq,
            },
            self.fq_threshold,
        )
        record.payload["output_dir"] = "dry_run"
        record.payload["evaluator_json_path"] = "dry_run"
        self._append_ledger(
            record,
            CandidateSpec.from_dict(record.canonical_spec),
            Path("dry_run"),
            "dry_run",
        )
        return record

    def _train(self, spec_path: Path, trial: Path, max_steps: int, seed: int) -> None:
        save_steps = min(self.recipe.get("checkpoint_steps", [max(max_steps, 1)]))
        command = [
            "accelerate",
            "launch",
            "--config_file",
            "configs/accelerate/default_config.yaml",
            "--main_process_port",
            str(_free_port()),
            "src/train.py",
            "--config-name=unlearn.yaml",
            f"experiment={self.recipe['experiment']}",
            "trainer=MOGPU",
            f"trainer.method_args.candidate_spec_path={spec_path}",
            f"task_name=mogpu_{spec_path.stem}_{max_steps}_{seed}",
            f"paths.output_dir={trial}",
            f"trainer.args.seed={seed}",
            "trainer.args.per_device_train_batch_size=4",
            "trainer.args.gradient_accumulation_steps=4",
            "trainer.args.ddp_find_unused_parameters=true",
            "trainer.args.gradient_checkpointing=true",
            "trainer.args.do_eval=false",
            "trainer.args.eval_strategy=no",
            f"retain_logs_path={self.retain_logs_path}",
        ]
        pretrained = self.recipe.get("pretrained_model_name_or_path")
        if pretrained:
            command.extend(
                [
                    f"model.model_args.pretrained_model_name_or_path={pretrained}",
                    f"model.tokenizer_args.pretrained_model_name_or_path={pretrained}",
                ]
            )
        if max_steps > 0:
            command.extend(
                [
                    f"trainer.args.max_steps={max_steps}",
                    "trainer.args.save_strategy=steps",
                    f"+trainer.args.save_steps={max(1, int(save_steps))}",
                ]
            )
        else:
            command.extend(
                ["trainer.args.max_steps=-1", "trainer.args.save_strategy=no"]
            )
        if self.recipe.get("num_train_epochs") is not None:
            command.append(
                f"trainer.args.num_train_epochs={self.recipe['num_train_epochs']}"
            )
        command.extend(self._split_overrides(for_eval=False))
        subprocess.run(command, check=True, cwd=self.repo_root)

    def _eval(self, model_path: Path, eval_dir: Path) -> Path:
        eval_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "python",
            "src/eval.py",
            f"experiment={self.recipe['eval_experiment']}",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"model.tokenizer_args.pretrained_model_name_or_path={model_path}",
            f"paths.output_dir={eval_dir}",
            f"retain_logs_path={self.retain_logs_path}",
        ]
        command.extend(self._split_overrides(for_eval=True))
        subprocess.run(command, check=True, cwd=self.repo_root)
        summary = eval_dir / "TOFU_SUMMARY.json"
        if not summary.is_file():
            raise FileNotFoundError(summary)
        return summary

    def _split_overrides(self, for_eval: bool) -> list[str]:
        forget = self.recipe.get("forget_split")
        if not forget:
            return []
        retain = {
            "forget10": ("retain90", "holdout10"),
            "forget05": ("retain95", "holdout05"),
            "forget01": ("retain99", "holdout01"),
        }.get(forget, (None, None))
        overrides = [f"forget_split={forget}"]
        if retain[0]:
            overrides.append(f"holdout_split={retain[1]}")
            if not for_eval:
                overrides.append(f"retain_split={retain[0]}")
        return overrides

    def _append_ledger(
        self,
        record: CandidateRecord,
        spec: CandidateSpec,
        trial: Path,
        summary: str,
    ) -> None:
        Ledger(self.output_dir, record.tier).append(
            {
                "candidate_hash": record.candidate_hash,
                "parent_hashes": record.parent_hashes,
                "generation": record.generation,
                "operator": record.operator,
                "tier": record.tier,
                "stage": record.stage,
                "status": record.status,
                "output_dir": str(trial),
                "evaluator_json_path": str(summary),
                "fq_feasible": record.fq_feasible,
                "forget_quality": record.payload.get("forget_quality"),
                "objectives": record.objectives,
                "explanation_card": render_card(spec),
            }
        )


def write_spec(spec: CandidateSpec, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not math.isfinite(sum(spec.to_dict()["weights"])):
        raise ValueError("non-finite weights")
