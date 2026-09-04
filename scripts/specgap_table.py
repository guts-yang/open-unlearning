#!/usr/bin/env python3
"""Build the E0 SpecGap audit table and optionally join existing OU dimensions."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


OU_DIMENSIONS = ("Mem", "Priv", "Utility", "Agg")


def read_jsonl(path):
    records = {}
    if not Path(path).is_file():
        return records
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["name"]] = record
    return records


def format_number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def write_atomic(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_record(method, repo_id, result, ou_record):
    name = repo_id.rsplit("/", 1)[-1]
    forget = result["splits"]["forget10"]["summary"]
    retain = result["splits"]["retain90"]["summary"]
    comparison = result["comparison"]
    record = {
        "name": name,
        "repo_id": repo_id,
        "method": method,
        "created_at": result["created_at"],
        "forget": forget,
        "retain": retain,
        "mean_difference": comparison["mean_difference"],
        "cohens_d": comparison["cohens_d"],
        "separated": (
            comparison["mean_difference"] > 0 and comparison["cohens_d"] > 0.8
        ),
        "ou_table3": None,
    }
    if ou_record is not None:
        record["ou_table3"] = {
            dimension: ou_record.get(dimension) for dimension in OU_DIMENSIONS
        }
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Generate SpecGap E0 JSONL and Markdown summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/specgap/e0_tofu_forget10.yaml")
    parser.add_argument(
        "--result-dir",
        default=os.environ.get("SPECGAP_E0_DIR", "/root/autodl-tmp/saves/specgap/e0"),
    )
    parser.add_argument("--ou-runs", default="results/ou_table3_runs.jsonl")
    parser.add_argument("--jsonl", default="results/specgap_e0.jsonl")
    parser.add_argument("--markdown", default="results/specgap_e0.md")
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ou_records = read_jsonl(args.ou_runs)
    records = []
    missing = []
    for checkpoint in config["checkpoints"]:
        method = checkpoint["method"]
        repo_id = checkpoint["repo_id"]
        name = repo_id.rsplit("/", 1)[-1]
        result_path = Path(args.result_dir) / f"{name}.json"
        if not result_path.is_file():
            missing.append((method, repo_id))
            continue
        with result_path.open() as handle:
            result = json.load(handle)
        if result.get("metric") != "SpecGap":
            raise ValueError(f"Unexpected metric in {result_path}")
        records.append(build_record(method, repo_id, result, ou_records.get(name)))

    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    write_atomic(args.jsonl, jsonl)

    lines = [
        "# SpecGap E0 审计（TOFU forget10 · Llama-3.2-1B-Instruct）",
        "",
        "forget10 全量；retain90 使用 seed=0 等量抽样。OU 四维只关联已有"
        " `results/ou_table3_runs.jsonl` 记录，不触发补评。",
        "",
        "| Method | SpecGap_f (95% CI) | SpecGap_r (95% CI) | Δ(f-r) | Cohen's d | 分离 | Mem | Priv | Utility | Agg |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    by_method = {record["method"]: record for record in records}
    for checkpoint in config["checkpoints"]:
        method = checkpoint["method"]
        record = by_method.get(method)
        if record is None:
            lines.append(
                f"| {method} | pending | pending | — | — | — | — | — | — | — |"
            )
            continue
        forget = record["forget"]
        retain = record["retain"]
        forget_cell = (
            f"{forget['mean']:.4f} "
            f"[{forget['ci95'][0]:.4f}, {forget['ci95'][1]:.4f}]"
        )
        retain_cell = (
            f"{retain['mean']:.4f} "
            f"[{retain['ci95'][0]:.4f}, {retain['ci95'][1]:.4f}]"
        )
        ou = record["ou_table3"] or {}
        lines.append(
            f"| {method} | {forget_cell} | {retain_cell} | "
            f"{record['mean_difference']:.4f} | {record['cohens_d']:.3f} | "
            f"{'yes' if record['separated'] else 'no'} | "
            f"{format_number(ou.get('Mem'))} | {format_number(ou.get('Priv'))} | "
            f"{format_number(ou.get('Utility'))} | {format_number(ou.get('Agg'))} |"
        )
    lines.extend(
        [
            "",
            f"完成：{len(records)}/{len(config['checkpoints'])}；"
            f"待跑：{len(missing)}。",
            "",
        ]
    )
    write_atomic(args.markdown, "\n".join(lines))
    print(
        f"[done] completed={len(records)} missing={len(missing)} "
        f"jsonl={args.jsonl} markdown={args.markdown}"
    )


if __name__ == "__main__":
    main()
