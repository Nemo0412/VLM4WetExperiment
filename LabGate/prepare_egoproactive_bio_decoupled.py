#!/usr/bin/env python3
"""Build separate small-gate and oracle-gated expert evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_egoproactive_bio_labgate import build_records


# Each inner list is a synonym group. A guidance response covers a group when it
# contains at least one phrase. These rubrics are audit/scoring data, never prompts.
SEMANTIC_RUBRICS = {
    "d01": [["take", "retrieve", "remove"], ["flask"], ["incubator"]],
    "d03": [["uv"], ["turn off", "switch off", "turned off"]],
    "d05": [["glove"], ["put on", "wear"], ["disinfect", "alcohol", "sanitize"]],
    "d07": [["sash"], ["lower", "working height", "proper height"]],
    "d09": [["pbs"], ["add", "rinse"]],
    "d10": [["pbs"], ["add", "rinse"], ["dry", "exposed", "immediately"]],
    "d12": [["trypsin"], ["add"]],
    "d14": [["shake", "shaking"], ["avoid", "do not", "don't"], ["gently", "rock", "tap"]],
    "d16": [["incubator"], ["2 minute", "two minute"]],
    "d18": [["growth medium", "complete medium"], ["neutralize"], ["two volume", "2 volume"]],
    "d19": [["growth medium", "complete medium"], ["neutralize"], ["add"]],
    "d21": [["transfer"], ["new flask"]],
    "d23": [["label"], ["flask"]],
    "d24": [["incubator"], ["return", "place", "put"]],
}


def build_decoupled(dataset: Path) -> tuple[list[dict], list[dict]]:
    rows = build_records(dataset)
    gate_fields = [
        "id", "video", "frame_indices", "protocol", "asr_text",
        "gt_fire", "gt_reason", "gt_type", "phase", "pair_id", "interval_sec",
    ]
    gate_rows = [{key: row[key] for key in gate_fields} for row in rows]
    expert_rows = []
    for row in rows:
        if not row["gt_fire"]:
            continue
        expert_rows.append(
            {
                "id": row["id"],
                "video": row["video"],
                "frame_indices": row["frame_indices"],
                "protocol": row["protocol"],
                "asr_text": row["asr_text"],
                "oracle_reason": row["gt_reason"],
                "gt_type": row["gt_type"],
                "reference_guidance": row["reference_guidance"],
                "semantic_rubric": SEMANTIC_RUBRICS[row["id"]],
                "phase": row["phase"],
                "pair_id": row["pair_id"],
                "interval_sec": row["interval_sec"],
            }
        )
    if len(gate_rows) != 25 or len(expert_rows) != 14:
        raise ValueError("Expected 25 gate cases and 14 oracle-expert cases")
    if set(SEMANTIC_RUBRICS) != {row["id"] for row in expert_rows}:
        raise ValueError("Semantic rubrics do not match positive decisions")
    return gate_rows, expert_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gate_rows, expert_rows = build_decoupled(args.dataset)
    gate_path = args.out_dir / "gate_eval.jsonl"
    expert_path = args.out_dir / "expert_oracle_eval.jsonl"
    write_jsonl(gate_path, gate_rows)
    write_jsonl(expert_path, expert_rows)
    metadata = {
        "annotation_sha256": hashlib.sha256(
            (args.dataset / "egoproactive/egoproactive_bio_val.jsonl").read_bytes()
        ).hexdigest(),
        "gate": {
            "n": 25,
            "model_inputs": ["video", "frame_indices", "protocol", "asr_text"],
            "scoring_only": ["gt_fire", "gt_reason", "gt_type", "phase", "pair_id", "interval_sec"],
            "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        },
        "expert": {
            "n": 14,
            "model_inputs": ["video", "frame_indices", "protocol", "asr_text", "oracle_reason"],
            "oracle_input": "oracle_reason is the reason a correct small gate should pass",
            "scoring_only": ["gt_type", "reference_guidance", "semantic_rubric", "phase", "pair_id", "interval_sec"],
            "sha256": hashlib.sha256(expert_path.read_bytes()).hexdigest(),
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
