#!/usr/bin/env python3
"""Map EgoProactive-Bio decisions to the native LabGate evaluation schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from egoproactive_bio import CELL_PASSAGING_PROTOCOL
from prepare_eval import window_indices


SAFETY_PAIRS = {"uv_on", "no_gloves"}


def target_type(decision: dict) -> tuple[str, str]:
    answer = decision["answer"]
    if answer == "$silent$":
        return "none", "none"
    if not answer.startswith("$interrupt$"):
        raise ValueError(f"Unknown answer format for {decision['decision_id']}")
    if decision["phase"] == "error":
        return (
            ("safety", "safety")
            if decision.get("pair_id") in SAFETY_PAIRS
            else ("action_error", "action_error")
        )
    return "assistant", "next_step"


def build_records(dataset: Path, all_frames: int = 16) -> list[dict]:
    compact_path = dataset / "egoproactive/egoproactive_bio_val.jsonl"
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    decisions = json.loads(
        (dataset / "metadata/composition.json").read_text(encoding="utf-8")
    )["decisions"]
    if not (
        len(decisions)
        == len(compact["answers"])
        == len(compact["video_intervals"])
        == len(compact["task"])
        == 25
    ):
        raise ValueError("Expected 25 aligned Bio decisions")
    video = dataset / "egoproactive/val" / compact["video_path"]
    rows = []
    for i, decision in enumerate(decisions):
        interval = compact["video_intervals"][i]
        if decision["target_interval_sec"] != interval:
            raise ValueError(f"Interval mismatch at {decision['decision_id']}")
        if decision["answer"] != compact["answers"][i]:
            raise ValueError(f"Answer mismatch at {decision['decision_id']}")
        indices = window_indices(video, interval[0], interval[1], all_frames)
        if indices is None or len(indices) != all_frames:
            raise ValueError(f"Could not extract {all_frames} frames for {decision['decision_id']}")
        gt_type, gt_reason = target_type(decision)
        rows.append(
            {
                "id": decision["decision_id"],
                "video": str(video),
                "frame_indices": indices,
                "asr_text": "",
                "protocol": CELL_PASSAGING_PROTOCOL,
                "gt_type": gt_type,
                "gt_fire": gt_type != "none",
                "gt_reason": gt_reason,
                "reference_guidance": (
                    compact["answers"][i].removeprefix("$interrupt$")
                    if compact["answers"][i].startswith("$interrupt$")
                    else ""
                ),
                "phase": decision["phase"],
                "pair_id": decision.get("pair_id"),
                "task_for_audit_only": compact["task"][i],
                "interval_sec": interval,
                "note": "Mapped from immutable EgoProactive-Bio annotations; audit fields are not model input.",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-frames", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; use a new path")
    rows = build_records(args.dataset, args.all_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema": "native LabGate eval record + audit-only Bio fields",
        "n": len(rows),
        "by_gt_type": dict(Counter(row["gt_type"] for row in rows)),
        "by_gt_reason": dict(Counter(row["gt_reason"] for row in rows)),
        "annotation_sha256": hashlib.sha256(
            (args.dataset / "egoproactive/egoproactive_bio_val.jsonl").read_bytes()
        ).hexdigest(),
        "label_mapping": {
            "silent": "none",
            "error/uv_on or error/no_gloves": "safety",
            "other error": "action_error",
            "other interrupt": "assistant with gate reason next_step",
        },
        "model_inputs": ["video/frame_indices", "protocol", "asr_text"],
        "audit_only_not_model_inputs": [
            "gt_type", "gt_fire", "gt_reason", "reference_guidance", "phase",
            "pair_id", "task_for_audit_only", "interval_sec", "note",
        ],
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
