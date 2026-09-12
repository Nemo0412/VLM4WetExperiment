#!/usr/bin/env python3
"""Build Bio evaluation records with the dataset-provided history and current task."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from prepare_egoproactive_bio_decoupled import SEMANTIC_RUBRICS
from prepare_egoproactive_bio_labgate import build_records


def normalize_history(history: list[dict], *, strip_control_tokens: bool) -> list[dict[str, str]]:
    normalized = []
    for turn in history:
        text = str(turn.get("text", ""))
        if strip_control_tokens:
            text = re.sub(r"\$(?:interrupt|silent)\$", "", text, flags=re.I).strip()
        if not text:
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role not in {"user", "assistant"}:
            role = "user"
        normalized.append({"role": role, "content": text})
    return normalized


def official_history(
    history: list[dict], *, strip_control_tokens: bool, max_history_turns: int = 4
) -> list[dict[str, str]]:
    """Match WearableAI: keep the high-level query plus the latest prior turns."""
    normalized = normalize_history(history, strip_control_tokens=strip_control_tokens)
    if not normalized:
        return []
    query, later = normalized[0], normalized[1:]
    if max_history_turns == 0:
        later = []
    elif max_history_turns > 0:
        later = later[-max_history_turns:]
    return [query, *later]


def build_contextual_records(dataset: Path) -> tuple[list[dict], list[dict]]:
    native = build_records(dataset)
    sample = json.loads(
        (dataset / "egoproactive/egoproactive_bio_val.jsonl").read_text(encoding="utf-8")
    )
    if not len(native) == len(sample["dialog"]) == len(sample["task"]) == 25:
        raise ValueError("Expected 25 aligned native/history/task records")
    gate_rows = []
    expert_rows = []
    for row, history, current_step in zip(native, sample["dialog"], sample["task"]):
        shared = {
            "id": row["id"], "video": row["video"],
            "frame_indices": row["frame_indices"], "protocol": row["protocol"],
            "asr_text": row["asr_text"], "current_step": current_step,
            "history_official_raw": official_history(
                history, strip_control_tokens=False
            ),
            "history_official_clean": official_history(
                history, strip_control_tokens=True
            ),
            "history_full_raw": normalize_history(
                history, strip_control_tokens=False
            ),
            "history_full_clean": normalize_history(
                history, strip_control_tokens=True
            ),
            "gt_type": row["gt_type"], "gt_fire": row["gt_fire"],
            "gt_reason": row["gt_reason"], "phase": row["phase"],
            "pair_id": row["pair_id"], "interval_sec": row["interval_sec"],
        }
        gate_rows.append(shared)
        if row["gt_fire"]:
            expert_rows.append({
                **shared,
                "oracle_reason": row["gt_reason"],
                "reference_guidance": row["reference_guidance"],
                "semantic_rubric": SEMANTIC_RUBRICS[row["id"]],
            })
    if len(expert_rows) != 14:
        raise ValueError("Expected 14 oracle-positive expert records")
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
    gate, expert = build_contextual_records(args.dataset)
    gate_path = args.out_dir / "gate_contextual.jsonl"
    expert_path = args.out_dir / "expert_contextual.jsonl"
    write_jsonl(gate_path, gate)
    write_jsonl(expert_path, expert)
    metadata = {
        "annotation_sha256": hashlib.sha256(
            (args.dataset / "egoproactive/egoproactive_bio_val.jsonl").read_bytes()
        ).hexdigest(),
        "n_gate": len(gate), "n_expert": len(expert),
        "shared_model_context": [
            "video/frame_indices", "protocol", "asr_text", "current_step",
            "history_official_raw/clean or history_full_raw/clean",
        ],
        "history_official_raw": "high-level query plus latest 4 prior turns, matching WearableAI, with control tokens",
        "history_official_clean": "same official history with $interrupt$/$silent$ removed",
        "history_full_raw": "exact full cumulative dialog before this decision, including control tokens",
        "history_full_clean": "same full cumulative dialog with control tokens removed",
        "current_step": "exact aligned task[i] supplied by the dataset",
        "warning": "Some task entries explicitly contain error/recovery wording; report separately from vision-only runs.",
        "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "expert_sha256": hashlib.sha256(expert_path.read_bytes()).hexdigest(),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
