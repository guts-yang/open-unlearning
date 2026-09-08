"""LeakRU QA loader and Dataset. Real benchmark files are not bundled."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from open_unlearning_plugins.leakru.scoring import LOGIC_TYPES, REQUIRED_FIELDS, SPLITS

DEFAULT_DATA_DIR = Path("/root/autodl-tmp/data/leakru")
SUBSET_FILENAMES = {
    "mquake": ("mquake.jsonl", "mquake.json", "leakru_mquake.jsonl", "leakru_mquake.json"),
    "books": ("books.jsonl", "books.json", "leakru_books.jsonl", "leakru_books.json"),
}


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"LeakRU file is empty: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Each LeakRU JSONL row must be an object ({path}:{line_no})")
            rows.append(obj)
        return rows
    payload = json.loads(text)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError(f"LeakRU JSON must be a list or {{data: list}} ({path})")
    rows = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            raise ValueError(f"LeakRU JSON item {i} is not an object ({path})")
        rows.append(obj)
    return rows


def _validate_row(row: dict[str, Any], index: int, source: Path) -> dict[str, Any]:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise ValueError(
            f"LeakRU row {index} in {source} missing fields {missing}. "
            f"Required: {list(REQUIRED_FIELDS)}"
        )
    logic = str(row["logic_type"]).strip()
    if logic not in LOGIC_TYPES:
        raise ValueError(
            f"LeakRU row {index} in {source} has logic_type={logic!r}; "
            f"expected one of {LOGIC_TYPES}"
        )
    split = str(row["split"]).strip()
    if split not in SPLITS:
        raise ValueError(
            f"LeakRU row {index} in {source} has split={split!r}; expected one of {SPLITS}"
        )
    aliases = row["aliases"]
    if aliases is None:
        aliases = []
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        raise ValueError(f"LeakRU row {index} aliases must be a list of strings ({source})")
    out = dict(row)
    out["question"] = str(row["question"])
    out["answer"] = str(row["answer"])
    out["aliases"] = [str(a) for a in aliases]
    out["logic_type"] = logic
    out["split"] = split
    if "key_tokens" in out and out["key_tokens"] is not None:
        kt = out["key_tokens"]
        if isinstance(kt, str):
            kt = [kt]
        out["key_tokens"] = [str(t) for t in kt]
    return out


def resolve_leakru_path(data_path: str | os.PathLike | None, subset: str = "mquake") -> Path:
    subset = subset.lower()
    if subset not in SUBSET_FILENAMES:
        raise ValueError(f"Unknown LeakRU subset {subset!r}; expected {tuple(SUBSET_FILENAMES)}")

    candidates: list[Path] = []
    if data_path not in (None, "", "null"):
        candidates.append(Path(str(data_path)).expanduser())
    env_dir = os.environ.get("LEAKRU_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(DEFAULT_DATA_DIR)

    tried: list[str] = []
    for cand in candidates:
        tried.append(str(cand))
        if cand.is_file():
            return cand
        if cand.is_dir():
            for name in SUBSET_FILENAMES[subset]:
                hit = cand / name
                tried.append(str(hit))
                if hit.is_file():
                    return hit
    raise FileNotFoundError(
        "LeakRU data not found. Place the official json/jsonl on the data disk and set "
        "eval.leakru.data_path or LEAKRU_DATA_DIR. "
        f"Looked at: {tried}. Expected files for subset={subset}: {SUBSET_FILENAMES[subset]}. "
        "Do not invent samples."
    )


def load_leakru(
    data_path: str | os.PathLike | None = None,
    subset: str = "mquake",
    per_type_limit: int | None = None,
    splits: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    path = resolve_leakru_path(data_path, subset=subset)
    raw = _read_records(path)
    rows = [_validate_row(row, i, path) for i, row in enumerate(raw)]
    if splits:
        allow = set(splits)
        rows = [r for r in rows if r["split"] in allow]
    if not rows:
        raise ValueError(f"LeakRU file {path} produced zero rows after filtering.")

    grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in LOGIC_TYPES}
    for row in rows:
        grouped[row["logic_type"]].append(row)

    missing_types = [k for k, v in grouped.items() if not v]
    if missing_types:
        counts = {k: len(v) for k, v in grouped.items()}
        raise ValueError(
            f"LeakRU data at {path} is missing logic_type(s) {missing_types}. "
            f"Counts: {counts}. Need all of {LOGIC_TYPES}."
        )

    limited: list[dict[str, Any]] = []
    limit = None if per_type_limit in (None, "null") else int(per_type_limit)
    for logic in LOGIC_TYPES:
        chunk = grouped[logic]
        if limit is not None:
            chunk = chunk[:limit]
        limited.extend(chunk)
    return limited


class LeakRUDataset:
    def __init__(
        self,
        tokenizer,
        template_args,
        data_path=None,
        subset="mquake",
        per_type_limit=None,
        splits=None,
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=True,
        hf_args=None,
        **_kwargs,
    ):
        super().__init__()
        if hf_args:
            data_path = hf_args.get("data_path", data_path)
            subset = hf_args.get("subset", subset)
            per_type_limit = hf_args.get("per_type_limit", per_type_limit)
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.max_length = max_length
        self.predict_with_generate = predict_with_generate
        self.question_key = question_key
        self.answer_key = answer_key
        rows = load_leakru(
            data_path=data_path,
            subset=subset,
            per_type_limit=per_type_limit,
            splits=splits,
        )
        from data.utils import add_dataset_index
        from datasets import Dataset as HFDataset

        self.data = add_dataset_index(HFDataset.from_list(rows))

    def __len__(self):
        return len(self.data)

    def row_at(self, index: int) -> dict[str, Any]:
        return self.data[int(index)]

    def _process_sample(self, question, answer, index=-1):
        from data.utils import preprocess_chat_instance

        tokenized = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [question],
            [answer],
            self.max_length,
            self.predict_with_generate,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
            "index": index,
        }

    def __getitem__(self, idx):
        row = self.data[idx]
        return self._process_sample(
            question=row[self.question_key],
            answer=row[self.answer_key],
            index=row["index"],
        )
