#!/usr/bin/env python3
"""Append one MUSE/WMDP run to results/muse_wmdp_runs.jsonl and refresh results/muse_wmdp.md.

Do not write ou_table3_runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
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

CODE_LINK = "https://github.com/locuslab/open-unlearning"

METHOD_META = {
    "GradAscent": ("Gradient Ascent", "经典遗忘基线"),
    "GradDiff": ("Gradient Difference", "Liu et al., 2022"),
    "NPO": ("Negative Preference Optimization", "Zhang et al., 2024"),
    "SimNPO": ("Simple Negative Preference Optimization", "Fan et al., 2024"),
    "RMU": ("Representation Misdirection for Unlearning", "Li et al., 2024"),
    "CEU": ("Cross-Entropy Unlearning", "仓库扩展方法"),
    "PDU": ("Primal-Dual Unlearning", "community/methods/PDU"),
    "DPO": ("Direct Preference Optimization", "Rafailov et al., 2023"),
    "UNDIAL": ("UNDIAL", "NAACL 2025"),
    "WGA": ("Weighted Gradient Ascent", "仓库扩展方法"),
    "SatImp": ("Saturated Implicit Misfit", "仓库扩展方法"),
    "Finetuned": ("Finetuned target（未遗忘）", "MUSE 官方对照日志"),
    "Retain": ("Retrain on retain only", "MUSE 官方对照日志"),
}

# docs/repro.md：KnowMem_Df / KnowMem_Dr / VerbMem_Df / PrivLeak
OFFICIAL_MUSE = {
    ("News", "GradAscent"): "0 / 0 / 0 / 52.11",
    ("News", "GradDiff"): "0.41 / 0.37 / 8.92e-03 / 93.23",
    ("News", "NPO"): "0.56 / 0.51 / 0.35 / -86.00",
    ("News", "SimNPO"): "0.54 / 0.51 / 0.36 / -86.11",
    ("News", "RMU"): "0.48 / 0.51 / 0.05 / 56.36",
    ("Books", "GradAscent"): "0 / 0 / 0 / -0.67",
    ("Books", "GradDiff"): "0.18 / 0.30 / 0.16 / -37.79",
    ("Books", "NPO"): "0.32 / 0.55 / 0.84 / -54.24",
    ("Books", "SimNPO"): "0.32 / 0.54 / 0.84 / -54.26",
    ("Books", "RMU"): "0.29 / 0.48 / 0.79 / -60.52",
}

MUSE_ORDER = [
    "GradAscent",
    "GradDiff",
    "NPO",
    "SimNPO",
    "RMU",
    "CEU",
    "PDU",
    "DPO",
    "UNDIAL",
    "WGA",
    "SatImp",
    "Finetuned",
    "Retain",
]

MUSE_WIDE_HEADER = (
    "| 基座名称 | SOTA | 代码链接 | 方法 | KnowMem_Df↓ | KnowMem_Dr↑ | "
    "VerbMem_Df↓ | PrivLeak→0 | 全称 | 年份/出处 | Batch | LR | Epoch | GPU型号×数量 |"
)
MUSE_WIDE_SEP = "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|"
WMDP_WIDE_HEADER = (
    "| 基座名称 | SOTA | 代码链接 | 方法 | Bio Acc↓ | MMLU↑ | Cyber Acc↓ | MT-Bench↑ | "
    "全称 | 年份/出处 | Batch | LR | Epoch | GPU型号×数量 |"
)
WMDP_WIDE_SEP = "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|"
HIST_MARK = "## 此前（V100）"


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:  # NaN
        return "NaN"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e3):
        return f"{x:.2e}"
    return f"{x:.4f}"


def load_jsonl() -> list[dict]:
    if not JSONL.exists():
        return []
    rows = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def upsert(rows: list[dict], run: dict) -> list[dict]:
    key = (run.get("bench"), run.get("split"), run.get("method"))
    kept = [
        r
        for r in rows
        if (r.get("bench"), r.get("split"), r.get("method")) != key
    ]
    kept.append(run)
    kept.sort(key=lambda r: (r.get("bench", ""), r.get("split", ""), r.get("method", "")))
    return kept


def write_jsonl(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def muse_base_name(split: str) -> str:
    return f"Llama-2-7B (MUSE-{split})"


def parse_gpu_batch(hyper: str) -> tuple[str, str]:
    h = hyper.replace("×", "x")
    if "A800" in hyper:
        gpu = "A800×2"
    elif "V100" in hyper:
        gpu = "V100×1"
    else:
        gpu = "—"
    if "4x4x2" in h:
        batch = "32 (4×4×2)"
    elif "1x8x2" in h:
        batch = "16 (1×8×2)"
    elif "1x32" in h:
        batch = "32 (1×32×1)"
    else:
        batch = "—"
    return gpu, batch


def muse_wide_row(r: dict) -> str:
    m = r.get("metrics") or {}
    method = r.get("method", "")
    split = r.get("split") or "News"
    full, year = METHOD_META.get(method, (method, "—"))
    gpu, batch = parse_gpu_batch(r.get("hyper") or "")
    return (
        "| {base} | {sota} | {code} | {method} | {fkm} | {rkm} | {vm} | {pl} | "
        "{full} | {year} | {batch} | {lr} | {ep} | {gpu} |"
    ).format(
        base=muse_base_name(split),
        sota=OFFICIAL_MUSE.get((split, method), "—"),
        code=CODE_LINK,
        method=method,
        fkm=fmt(m.get("forget_knowmem_ROUGE")),
        rkm=fmt(m.get("retain_knowmem_ROUGE")),
        vm=fmt(m.get("forget_verbmem_ROUGE")),
        pl=fmt(m.get("privleak")),
        full=full,
        year=year,
        batch=batch,
        lr="1e-5",
        ep="10",
        gpu=gpu,
    )


def write_md(rows: list[dict]) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# MUSE / WMDP 本机复现",
        "",
        f"更新时间：{now}",
        "",
        "口径见 `docs/zh/【3】muse-wmdp-repro.md`。**不进** `ou_table3_runs.jsonl`。",
        "",
        "SOTA 列为 `docs/repro.md` 官方对照（KnowMem_Df / KnowMem_Dr / VerbMem_Df / PrivLeak）；后四列为本机复现。",
        "",
        "## 本轮",
        "",
        "### MUSE",
        "",
        MUSE_WIDE_HEADER,
        MUSE_WIDE_SEP,
    ]
    muse = [r for r in rows if r.get("bench") == "MUSE"]
    muse.sort(
        key=lambda r: (
            r.get("split") or "",
            MUSE_ORDER.index(r.get("method")) if r.get("method") in MUSE_ORDER else 99,
        )
    )
    for r in muse:
        lines.append(muse_wide_row(r))
    if not muse:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "### WMDP",
        "",
        WMDP_WIDE_HEADER,
        WMDP_WIDE_SEP,
    ]
    wmdp = [r for r in rows if r.get("bench") == "WMDP"]
    for r in wmdp:
        m = r.get("metrics") or {}
        method = r.get("method", "")
        full, year = METHOD_META.get(method, (method, "—"))
        gpu, batch = parse_gpu_batch(r.get("hyper") or "")
        if batch == "—":
            batch = "16 (1×8×2)"
        lines.append(
            "| {base} | {sota} | {code} | {method} | {bio} | {mmlu} | {cyber} | {mt} | "
            "{full} | {year} | {batch} | {lr} | {ep} | {gpu} |".format(
                base="Zephyr-7B-β (WMDP-cyber)",
                sota="—",
                code=CODE_LINK,
                method=method,
                bio=fmt(m.get("wmdp_bio/acc") or m.get("wmdp-bio/acc")),
                mmlu=fmt(m.get("mmlu/acc") or m.get("mmlu/acc,none")),
                cyber=fmt(
                    m.get("wmdp_cyber/acc")
                    or m.get("wmdp-cyber/acc")
                    or m.get("wmdp_cyber/acc,none")
                ),
                mt=fmt(m.get("mt_bench") or m.get("mt-bench")),
                full=full,
                year=year,
                batch=batch,
                lr="5e-5",
                ep="80 steps",
                gpu=gpu,
            )
        )
    if not wmdp:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.append("")
    if MD.exists():
        old = MD.read_text(encoding="utf-8")
        idx = old.find(HIST_MARK)
        if idx != -1:
            lines += [old[idx:].rstrip(), ""]
    MD.write_text("\n".join(lines), encoding="utf-8")


def extract_wmdp_metrics(summary: dict) -> dict:
    out = {}
    for k, v in summary.items():
        if not isinstance(k, str) or "stderr" in k.lower():
            continue
        lk = k.lower().replace("-", "_")
        if lk.startswith("wmdp_cyber/") and "acc" in lk:
            out["wmdp_cyber/acc"] = v
        elif lk.startswith("wmdp_bio/") and "acc" in lk:
            out["wmdp_bio/acc"] = v
        elif lk.startswith("mmlu/") and "acc" in lk:
            out.setdefault("mmlu/acc", v)
        elif "mt_bench" in lk:
            out["mt_bench"] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True, choices=["MUSE", "WMDP"])
    p.add_argument("--split", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--status", default="ok")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--ckpt", default="")
    p.add_argument("--hyper", default="")
    p.add_argument("--note", default="")
    args = p.parse_args()

    metrics = {}
    if args.summary and args.summary.exists():
        raw = json.loads(args.summary.read_text(encoding="utf-8"))
        if args.bench == "MUSE":
            metrics = {k: raw.get(k) for k in MUSE_KEYS}
            extra = {k: v for k, v in raw.items() if k not in MUSE_KEYS}
            if extra:
                metrics["_extra"] = extra
        else:
            metrics = extract_wmdp_metrics(raw)
            metrics["_raw_keys"] = sorted(raw.keys())

    run = {
        "bench": args.bench,
        "split": args.split,
        "method": args.method,
        "status": args.status,
        "ckpt": args.ckpt,
        "hyper": args.hyper,
        "note": args.note,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    rows = upsert(load_jsonl(), run)
    write_jsonl(rows)
    write_md(rows)
    print(f"updated {JSONL} and {MD} ({args.bench} {args.split} {args.method} {args.status})")


if __name__ == "__main__":
    main()
