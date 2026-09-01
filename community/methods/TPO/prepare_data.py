"""Convert the pre-annotated TOFU forget splits (UW/GW common-words spans) from the
official Unlearning-TPO repository into JSON files usable by this repo's data pipeline.

Source repo: https://github.com/guts-yang/Unlearning-TPO
(paper: "Not All Tokens Are Meant to Be Forgotten", AAAI 2026, arXiv:2506.03142)

The official repo stores the annotated forget splits as HuggingFace `load_from_disk`
(arrow) datasets under `TOFU/data/forget{01,05,10}_with_common_words_{bert,gpt}`.
Each example contains:
  - question: str
  - answer: str
  - common_words: list[str]  (each item is a JSON string {"word": ..., "start": ..., "end": ...}
                              with character-level spans relative to `answer`)
  - target_words: list[str]  (gpt variant only, kept for reference; not used by TPO loss)

Usage:
    python community/methods/TPO/prepare_data.py --src /usr/local/Unlearning-TPO

This script is only needed once; the converted JSON files are committed under
`community/methods/TPO/data/` so that `run.sh` works without cloning the official repo.
"""

import argparse
import json
import os

from datasets import load_from_disk

SPLITS = ["forget01", "forget05", "forget10"]
CLASSIFIERS = ["bert", "gpt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default="/usr/local/Unlearning-TPO",
        help="Path to a local clone of https://github.com/guts-yang/Unlearning-TPO",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in SPLITS:
        for classifier in CLASSIFIERS:
            src_dir = os.path.join(
                args.src, "TOFU", "data", f"{split}_with_common_words_{classifier}"
            )
            if not os.path.isdir(src_dir):
                raise FileNotFoundError(
                    f"{src_dir} not found. Clone the official repo first:\n"
                    f"  git clone https://github.com/guts-yang/Unlearning-TPO.git {args.src}"
                )
            ds = load_from_disk(src_dir)
            records = []
            for example in ds:
                record = {
                    "question": example["question"],
                    "answer": example["answer"],
                    # keep the raw JSON-string list structure from the official release
                    "common_words": list(example["common_words"]),
                }
                if "target_words" in example:
                    record["target_words"] = list(example["target_words"])
                records.append(record)

            out_path = os.path.join(
                args.out_dir, f"{split}_with_common_words_{classifier}.json"
            )
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"wrote {out_path} ({len(records)} examples)")


if __name__ == "__main__":
    main()
