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


def lookup_ou(ou_records, names):
    for name in names:
        if name and name in ou_records:
            return ou_records[name]
    return None


def expand_checkpoints(checkpoints, result_dir=None):
    """Preferred local-best first, then the original official anchor if different."""
    result_dir = Path(result_dir) if result_dir else None
    rows = []
    seen = set()
    for item in checkpoints:
        local_name = item.get("local_name")
        preferred_name = item["repo_id"].rsplit("/", 1)[-1]
        local_path = item.get("local_path")
        local_json = (
            result_dir / f"{local_name}.json" if result_dir and local_name else None
        )
        local_weights = (
            bool(local_path) and (Path(local_path) / "config.json").is_file()
        )
        if local_name and (local_weights or (local_json and local_json.is_file())):
            best_name, join_names = local_name, [local_name, preferred_name]
        else:
            best_name, join_names = preferred_name, [preferred_name, local_name]
        candidates = [
            {
                "kind": "local-best",
                "method": item["method"],
                "repo_id": item["repo_id"],
                "name": best_name,
                "join_names": join_names,
            }
        ]
        official = item.get("official_repo_id")
        if official and official != item["repo_id"]:
            candidates.append(
                {
                    "kind": "official-anchor",
                    "method": item["method"],
                    "repo_id": official,
                    "name": official.rsplit("/", 1)[-1],
                    "join_names": [official.rsplit("/", 1)[-1]],
                }
            )
        for candidate in candidates:
            if candidate["name"] in seen:
                continue
            seen.add(candidate["name"])
            rows.append(candidate)
    return rows


def build_record(checkpoint, result, ou_record):
    forget = result["splits"]["forget10"]["summary"]
    retain = result["splits"]["retain90"]["summary"]
    comparison = result["comparison"]
    record = {
        "name": checkpoint["name"],
        "repo_id": checkpoint["repo_id"],
        "method": checkpoint["method"],
        "kind": checkpoint["kind"],
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
    checkpoints = expand_checkpoints(config["checkpoints"], args.result_dir)
    records = []
    missing = []
    for checkpoint in checkpoints:
        result_path = Path(args.result_dir) / f"{checkpoint['name']}.json"
        if not result_path.is_file():
            missing.append((checkpoint["method"], checkpoint["name"]))
            continue
        with result_path.open() as handle:
            result = json.load(handle)
        if result.get("metric") != "SpecGap":
            raise ValueError(f"Unexpected metric in {result_path}")
        records.append(
            build_record(
                checkpoint,
                result,
                lookup_ou(ou_records, checkpoint["join_names"]),
            )
        )

    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    write_atomic(args.jsonl, jsonl)

    lines = [
        "# SpecGap E0 审计（TOFU forget10 · Llama-3.2-1B-Instruct）",
        "",
        "每个方法优先审计 `ou_table3.md` 本机 Agg 最高点（local-best），"
        "再保留原官方锚点。forget10 全量；retain90 使用 seed=0 等量抽样。"
        "OU 四维只关联已有 `results/ou_table3_runs.jsonl` 记录，不触发补评。",
        "",
        "| Method | Kind | SpecGap_f (95% CI) | SpecGap_r (95% CI) | Δ(f-r) | Cohen's d | 分离 | Mem | Priv | Utility | Agg |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    by_name = {record["name"]: record for record in records}
    for checkpoint in checkpoints:
        method = checkpoint["method"]
        record = by_name.get(checkpoint["name"])
        if record is None:
            lines.append(
                f"| {method} | {checkpoint['kind']} | pending | pending | — | — | — | — | — | — | — |"
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
            f"| {method} | {record['kind']} | {forget_cell} | {retain_cell} | "
            f"{record['mean_difference']:.4f} | {record['cohens_d']:.3f} | "
            f"{'yes' if record['separated'] else 'no'} | "
            f"{format_number(ou.get('Mem'))} | {format_number(ou.get('Priv'))} | "
            f"{format_number(ou.get('Utility'))} | {format_number(ou.get('Agg'))} |"
        )
    lines.extend(
        [
            "",
            f"完成：{len(records)}/{len(checkpoints)}；待跑：{len(missing)}。",
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
