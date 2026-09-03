#!/usr/bin/env python3
"""Build the four-way FineBio zero-shot LabGate evaluation set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from protocols import PROTOCOL_NAMES, format_protocol

VID = Path("/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
MIS = Path("/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")

MISTAKE_CASES = [
    ("P06_03_02", 3, 200.0, 260.0, "missing sterile water wash"),
    ("P17_02_02", 2, 55.0, 120.0, "extra PBS wash"),
    ("P11_06_01", 6, 100.0, 180.0, "extra wash buffer"),
]

CORRECT = [
    ("P25_01_01", 1, 20.0, 40.0),
    ("P25_02_01", 2, 25.0, 45.0),
    ("P25_03_01", 3, 30.0, 55.0),
    ("P25_05_01", 5, 40.0, 70.0),
    ("P26_01_01", 1, 15.0, 35.0),
    ("P26_03_01", 3, 40.0, 70.0),
    ("P27_05_01", 5, 50.0, 80.0),
    ("P28_01_01", 1, 18.0, 35.0),
    ("P28_06_01", 6, 40.0, 80.0),
    ("P32_06_01", 6, 50.0, 90.0),
]

ASSISTANT_UTTERANCES = [
    "What should I do next according to the protocol?",
    "Which buffer do I add in this step?",
    "Have I finished the current step yet?",
    "Which instrument should I use now?",
    "Remind me of the remaining steps.",
    "How long should I vortex?",
    "Is the magnetic rack the next instrument?",
    "Can you summarize the current protocol step?",
    "Do I need sterile water after ethanol?",
    "Where should I put the supernatant?",
]

SAFETY_UTTERANCES = [
    "The centrifuge is spinning with the lid open.",
    "I just spilled 70 percent ethanol next to a hot plate.",
    "The PCR machine is smoking and smells like burning plastic.",
    "I got lysate in my eye, it burns.",
    "There is a small flame near the ethanol bottle.",
    "I am not wearing gloves and the lysate touched my hand.",
    "The spin column tube cracked and liquid is leaking.",
    "I knocked over the magnetic beads bottle.",
    "Someone turned on a Bunsen burner near the ethanol wash.",
    "The centrifuge is making a loud grinding noise.",
]

ERROR_UTTERANCES = [
    "I think I skipped the sterile water wash.",
    "I just did an extra PBS wash by mistake.",
    "Wait, I added ethanol before the wash buffer.",
    "I forgot to vortex before spindown.",
    "I put the tube in the wrong rack, not the magnetic rack.",
]


def window_indices(path: Path, t0: float, t1: float, count: int) -> list[int] | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if nframes <= 1:
        return None
    i0 = max(0, int(round(t0 * fps)))
    i1 = min(nframes - 1, int(round(t1 * fps)))
    if i1 <= i0:
        i1 = min(nframes - 1, i0 + max(count, 8))
    k = min(count, i1 - i0 + 1)
    start = max(i0, min((i0 + i1) // 2 - k // 2, i1 - k + 1))
    return list(range(start, start + k))


def make_record(
    sample_id: str,
    video: Path,
    indices: list[int],
    protocol_id: int,
    gt_type: str,
    asr_text: str,
    note: str,
) -> dict:
    return {
        "id": sample_id,
        "video": str(video),
        "frame_indices": indices,
        "asr_text": asr_text,
        "protocol": format_protocol(protocol_id),
        "protocol_id": protocol_id,
        "gt_type": gt_type,
        "gt_fire": gt_type != "none",
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="/scratch/ll5914/Labos/LabGate/data/eval_v2.jsonl"
    )
    parser.add_argument("--all-frames", type=int, default=16)
    args = parser.parse_args()
    rows = []

    for i, (trial, protocol_id, t0, t1) in enumerate(CORRECT):
        video = VID / f"{trial}.mp4"
        indices = window_indices(video, t0, t1, args.all_frames)
        if indices is None:
            continue
        rows.extend(
            [
                make_record(
                    f"none_{trial}",
                    video,
                    indices,
                    protocol_id,
                    "none",
                    "",
                    "correct execution, silence",
                ),
                make_record(
                    f"assistant_{trial}",
                    video,
                    indices,
                    protocol_id,
                    "assistant",
                    ASSISTANT_UTTERANCES[i],
                    "user question",
                ),
                make_record(
                    f"safety_{trial}",
                    video,
                    indices,
                    protocol_id,
                    "safety",
                    SAFETY_UTTERANCES[i],
                    "spoken hazard",
                ),
            ]
        )

    for trial, protocol_id, t0, t1, note in MISTAKE_CASES:
        video = MIS / f"{trial}.mp4"
        indices = window_indices(video, t0, t1, args.all_frames)
        if indices is not None:
            rows.append(
                make_record(
                    f"error_{trial}",
                    video,
                    indices,
                    protocol_id,
                    "action_error",
                    "",
                    note,
                )
            )

    for i, (trial, protocol_id, t0, t1) in enumerate(CORRECT[:5]):
        video = VID / f"{trial}.mp4"
        indices = window_indices(video, t0, t1, args.all_frames)
        if indices is not None:
            rows.append(
                make_record(
                    f"error_asr_{trial}",
                    video,
                    indices,
                    protocol_id,
                    "action_error",
                    ERROR_UTTERANCES[i],
                    "spoken self-report of protocol mistake",
                )
            )

    mismatches = [
        ("P25_05_01", 1, 40.0, 70.0),
        ("P25_01_01", 5, 20.0, 40.0),
        ("P28_06_01", 5, 40.0, 80.0),
        ("P27_05_01", 1, 50.0, 80.0),
    ]
    for trial, protocol_id, t0, t1 in mismatches:
        video = VID / f"{trial}.mp4"
        indices = window_indices(video, t0, t1, args.all_frames)
        if indices is not None:
            rows.append(
                make_record(
                    f"error_mismatch_{trial}_P{protocol_id}",
                    video,
                    indices,
                    protocol_id,
                    "action_error",
                    "",
                    f"video {trial} vs protocol {protocol_id} "
                    f"({PROTOCOL_NAMES[protocol_id]})",
                )
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print({"n": len(rows), "by_type": dict(Counter(r["gt_type"] for r in rows))})


if __name__ == "__main__":
    main()
