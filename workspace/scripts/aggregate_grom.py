"""Aggregate GROM OpenUnlearning dumps. Paper Final formulas; no paper-number fill-in."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _md_table import replace_measured_section

GROM_HEADERS = [
    "run",
    "bench",
    "note",
    "1-Rouge",
    "1-Prob",
    "1-Extr",
    "MU",
    "Final",
    "Final_paper",
    "Time_min",
    "Time_paper",
    "status",
]

GROM_PLACEHOLDERS = [
    {
        "run": "未复现",
        "bench": "TOFU-05-7B",
        "Final_paper": "0.79",
        "Time_paper": "1.8",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "bench": "TOFU-10-1B",
        "note": "官方基座 1B，非 BLADE 3B",
        "Final_paper": "0.75",
        "Time_paper": "0.5",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "bench": "MUSE-News",
        "Final_paper": "55.93",
        "Time_paper": "0.3",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "bench": "MUSE-Books",
        "note": "跟 yaml MUSE-Books_target",
        "Final_paper": "76.20",
        "Time_paper": "0.3",
        "status": "未复现",
    },
    {
        "run": "未复现",
        "bench": "WMDP-Bio",
        "note": "Bio forget 门控",
        "Final_paper": "0.63",
        "Time_paper": "0.2",
        "status": "未复现",
    },
]


def _nested_get(obj, *keys, default=None):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _agg(data, key):
    val = data.get(key) if isinstance(data, dict) else None
    if isinstance(val, dict):
        return val.get("agg_value")
    return val


def infer_bench(path: Path) -> str | None:
    text = str(path).lower()
    if re.search(r"wmdp|bio", text):
        return "WMDP-Bio"
    if re.search(r"muse[_-]?news", text):
        return "MUSE-News"
    if re.search(r"muse[_-]?books", text):
        return "MUSE-Books"
    if re.search(r"tofu.*05|tofu5", text):
        return "TOFU-05-7B"
    if re.search(r"tofu.*10|tofu10", text):
        return "TOFU-10-1B"
    return None


def tofu_final(mu, rouge, prob, extr):
    if None in (mu, rouge, prob, extr):
        return None
    return (float(1.0 - rouge) + float(1.0 - prob) + float(1.0 - extr) + float(mu)) / 4.0


def _to_pct(x):
    x = float(x)
    return x * 100.0 if abs(x) <= 1.5 else x


def muse_final(verb, know_f, priv, know_r):
    if None in (verb, know_f, priv, know_r):
        return None
    vm, kf, pl, kr = map(_to_pct, (verb, know_f, priv, know_r))
    forget_avg = ((100.0 - vm) + (100.0 - kf) + (100.0 - abs(pl))) / 3.0
    return 0.5 * (kr + forget_avg)


def _empty_grid():
    rows = []
    for raw in GROM_PLACEHOLDERS:
        row = {h: "" for h in GROM_HEADERS}
        row.update(raw)
        rows.append(row)
    return rows


def _read_time(ckpt_dir: Path):
    meta = ckpt_dir / "grom_edit.json"
    if not meta.is_file() and ckpt_dir.parent.joinpath("grom_edit.json").is_file():
        meta = ckpt_dir.parent / "grom_edit.json"
    if not meta.is_file():
        return None
    data = json.loads(meta.read_text())
    return data.get("elapsed_min")


def _apply(rows, bench, measured, run):
    for row in rows:
        if row["bench"] == bench and row["status"] == "未复现":
            row["run"] = run
            for key, val in measured.items():
                if val not in (None, ""):
                    row[key] = val
            row["status"] = "实测"
            return
    extra = {h: "" for h in GROM_HEADERS}
    extra.update({"run": run, "bench": bench or "", "status": "实测", **measured})
    rows.append(extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saves-root", type=Path, default=Path("workspace/saves/unlearn")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("workspace/results/grom_table.md")
    )
    args = parser.parse_args()
    rows = _empty_grid()
    if args.saves_root.exists():
        for path in sorted(args.saves_root.glob("**/TOFU_EVAL.json")):
            data = json.loads(path.read_text())
            rouge = _agg(data, "forget_Q_A_ROUGE")
            prob = _agg(data, "forget_Q_A_Prob")
            extr = _agg(data, "extraction_strength")
            mu = _agg(data, "model_utility")
            bench = infer_bench(path) or "TOFU-10-1B"
            measured = {
                "1-Rouge": None if rouge is None else 1.0 - float(rouge),
                "1-Prob": None if prob is None else 1.0 - float(prob),
                "1-Extr": None if extr is None else 1.0 - float(extr),
                "MU": mu,
                "Final": tofu_final(mu, rouge, prob, extr),
                "Time_min": _read_time(path.parent),
            }
            _apply(rows, bench, measured, str(path.parent))
        for path in sorted(args.saves_root.glob("**/MUSE_SUMMARY.json")):
            data = json.loads(path.read_text())
            bench = infer_bench(path)
            verb = _nested_get(data, "forget_verbmem_ROUGE", "agg_value")
            if verb is None:
                verb = data.get("forget_verbmem_ROUGE")
            know_f = _nested_get(data, "forget_knowmem_ROUGE", "agg_value")
            if know_f is None:
                know_f = data.get("forget_knowmem_ROUGE")
            priv = _nested_get(data, "privleak", "agg_value")
            if priv is None:
                priv = data.get("privleak")
            know_r = _nested_get(data, "retain_knowmem_ROUGE", "agg_value")
            if know_r is None:
                know_r = data.get("retain_knowmem_ROUGE")
            measured = {
                "Final": muse_final(verb, know_f, priv, know_r),
                "Time_min": _read_time(path.parent),
                "note": f"vm={verb} kf={know_f} pl={priv} kr={know_r}",
            }
            _apply(rows, bench or "MUSE-News", measured, str(path.parent))
        for path in sorted(args.saves_root.glob("**/LMEval_SUMMARY.json")):
            data = json.loads(path.read_text())
            acc = None
            mmlu = None
            for key, val in data.items():
                lk = str(key).lower()
                if "wmdp_bio" in lk or lk == "wmdp_bio":
                    acc = val.get("acc,none", val.get("acc")) if isinstance(val, dict) else val
                if lk == "mmlu" or lk.endswith("/mmlu"):
                    mmlu = val.get("acc,none", val.get("acc")) if isinstance(val, dict) else val
            if acc is None and "acc" in data:
                acc = data["acc"]
            one_minus = None if acc is None else 1.0 - float(acc)
            final = (
                None
                if one_minus is None or mmlu is None
                else 0.5 * (float(one_minus) + float(mmlu))
            )
            measured = {
                "1-Rouge": one_minus,
                "MU": mmlu,
                "Final": final,
                "Time_min": _read_time(path.parent),
                "note": "1-Rouge column holds 1-AccBio; MU holds MMLU",
            }
            _apply(rows, "WMDP-Bio", measured, str(path.parent))
    replace_measured_section(args.out, GROM_HEADERS, rows)


if __name__ == "__main__":
    main()
