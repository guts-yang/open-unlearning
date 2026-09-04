#!/usr/bin/env python3
"""Append one MUSE/WMDP eval to results/muse_wmdp_runs.jsonl and refresh results/muse_wmdp.md.

Never writes ou_table3_runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import fcntl
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
JSONL = RESULTS / "muse_wmdp_runs.jsonl"
MD = RESULTS / "muse_wmdp.md"

MUSE_KEYS = [
    "forget_knowmem_ROUGE",
    "retain_knowmem_ROUGE",
    "forget_verbmem_ROUGE",
    "privleak",
    "extraction_strength",
]


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    if all(k in data for k in ("forget_knowmem_ROUGE", "privleak")) and not any(
        isinstance(v, dict) and "agg_value" in v for v in data.values()
    ):
        return data
    out = {}
    for k, v in data.items():
        if isinstance(v, dict) and "agg_value" in v:
            out[k] = v["agg_value"]
        elif isinstance(v, (int, float)):
            out[k] = v
    return out


def _hm_knowmem_privleak(metrics: dict) -> float | None:
    km = metrics.get("forget_knowmem_ROUGE")
    pl = metrics.get("privleak")
    if km is None or pl is None:
        return None
    # PrivLeak → 0 is better; map |privleak|/100 into a [0,1] closeness score.
    priv = max(0.0, 1.0 - min(abs(float(pl)) / 100.0, 1.0))
    km = max(float(km), 1e-12)
    priv = max(priv, 1e-12)
    return 2 * km * priv / (km + priv)


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if abs(x) >= 100 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.3g}"
        return f"{x:.4f}"
    return str(x)


def append_run(rec: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_path = JSONL.with_suffix(".jsonl.lock")
    with open(lock_path, "a", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        rows = []
        if JSONL.exists():
            for line in JSONL.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        key = (rec.get("benchmark"), rec.get("data_split"), rec.get("method"), rec.get("task_name"))
        rows = [
            r
            for r in rows
            if (r.get("benchmark"), r.get("data_split"), r.get("method"), r.get("task_name")) != key
        ]
        rows.append(rec)
        JSONL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        write_md(rows)


def write_md(rows: list[dict]) -> None:
    muse = [r for r in rows if r.get("benchmark") == "muse"]
    wmdp = [r for r in rows if r.get("benchmark") == "wmdp"]
    lines = [
        "# MUSE / WMDP 复现结果",
        "",
        "口径与 TOFU Table 3 四维不同，**不进** `ou_table3_runs.jsonl`。",
        "",
        "## MUSE",
        "",
        "| split | method | forget KnowMem | retain KnowMem | VerbMem | PrivLeak | ES | HM(KM,Priv≈0) | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(muse, key=lambda x: (x.get("data_split", ""), x.get("method", ""))):
        m = r.get("metrics") or {}
        lines.append(
            "| {split} | {method} | {fk} | {rk} | {vm} | {pl} | {es} | {hm} | {note} |".format(
                split=r.get("data_split"),
                method=r.get("method"),
                fk=_fmt(m.get("forget_knowmem_ROUGE")),
                rk=_fmt(m.get("retain_knowmem_ROUGE")),
                vm=_fmt(m.get("forget_verbmem_ROUGE")),
                pl=_fmt(m.get("privleak")),
                es=_fmt(m.get("extraction_strength")),
                hm=_fmt(r.get("hm_knowmem_privleak")),
                note=r.get("note") or r.get("status") or "",
            )
        )
    lines += [
        "",
        "## WMDP-cyber",
        "",
        "| method | wmdp_cyber acc | mmlu acc | note |",
        "|---|---:|---:|---|",
    ]
    for r in sorted(wmdp, key=lambda x: x.get("method", "")):
        m = r.get("metrics") or {}
        acc = m.get("wmdp_cyber/acc") or m.get("wmdp_cyber/acc,none")
        mmlu = m.get("mmlu/acc") or m.get("mmlu/acc,none")
        if acc is None:
            for k, v in m.items():
                if "wmdp" in k and "acc" in k:
                    acc = v
                if k.startswith("mmlu/") and "acc" in k and "stderr" not in k:
                    mmlu = v
        lines.append(
            f"| {r.get('method')} | {_fmt(acc)} | {_fmt(mmlu)} | {r.get('note') or r.get('status') or ''} |"
        )
    lines.append("")
    MD.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--benchmark", required=True, choices=("muse", "wmdp"))
    p.add_argument("--method", required=True)
    p.add_argument("--data-split", required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--hyper", default="")
    p.add_argument("--note", default="")
    p.add_argument("--status", default="ok")
    args = p.parse_args()

    metrics = {}
    if args.summary.is_file():
        metrics = _load_summary(args.summary)
        # also accept sibling EVAL
    elif args.summary.with_name("MUSE_EVAL.json").is_file():
        metrics = _load_summary(args.summary.with_name("MUSE_EVAL.json"))
    elif args.summary.with_name("LMEval_SUMMARY.json").is_file():
        metrics = _load_summary(args.summary.with_name("LMEval_SUMMARY.json"))

    rec = {
        "benchmark": args.benchmark,
        "data_split": args.data_split,
        "method": args.method,
        "task_name": args.task_name,
        "ckpt": args.ckpt,
        "hyper": args.hyper,
        "metrics": metrics,
        "status": args.status,
        "note": args.note,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.benchmark == "muse":
        rec["hm_knowmem_privleak"] = _hm_knowmem_privleak(metrics)
    append_run(rec)
    print("wrote", JSONL, "and", MD)


if __name__ == "__main__":
    main()
