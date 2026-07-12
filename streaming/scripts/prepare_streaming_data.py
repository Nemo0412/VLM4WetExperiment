#!/usr/bin/env python3
"""Build FineBioStreaming chunk-level dataset for online protocol monitoring.

Each sample is a video prefix ending at chunk k:
  - label CONTINUE for all chunks before the first violation
  - label HALT(+error_type, reason) at the first violating chunk
  - later chunks are not emitted (stream would have stopped)

Synthetic corruptions (drop / insert / shuffle) reuse FineBio step timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROTOCOL_NAMES = {
    1: "Cell lysate collection (single PBS wash)",
    2: "Cell lysate collection (double PBS wash)",
    3: "Magnetic-bead DNA extraction (single ethanol wash)",
    4: "Magnetic-bead DNA extraction (double ethanol wash)",
    5: "PCR reaction setup with 8-tube strips",
    6: "Spin-column DNA extraction (two wash steps)",
    7: "Spin-column DNA extraction (three wash steps)",
}
PROTOCOL_IDS = sorted(PROTOCOL_NAMES.keys())
PROTO_TO_CLASS = {p: i for i, p in enumerate(PROTOCOL_IDS)}

REAL_MISTAKES = {
    "P06_03_02": ("missing_step", "One or more steps are missing: the sterile water wash is skipped."),
    "P11_06_01": ("redundant_step", "There are redundant steps: an extra wash buffer step is performed."),
    "P17_02_02": ("redundant_step", "There are redundant steps: an extra PBS wash is performed."),
    "P03_03_02": ("within_step_error", "A within-step operation is missing: no pipetting during suction."),
    "P03_04_02": ("within_step_error", "A within-step operation is missing: no pipetting during suction."),
    "P07_04_01": ("within_step_error", "A within-step operation is missing: the vortex step is skipped."),
    "P17_07_01": ("wrong_order", "Steps are out of order: dispense and detach spin column are swapped."),
    "P18_02_01": ("within_step_error", "A within-step operation is missing: the spindown step is skipped."),
    "P18_05_01": ("missing_step", "The last step is missing (due to missing frames)."),
    "P24_07_01": ("missing_step", "A step is forgotten: dispensing after the second-to-last spindown is skipped."),
    "P28_06_02": ("wrong_order", "Steps are out of order: dispense and detach spin column are swapped."),
}

HALT_LABELS = {
    "continue": 0,
    "missing_step": 1,
    "redundant_step": 2,
    "wrong_order": 3,
    "within_step_error": 4,
}


def humanize(task: str) -> str:
    s = task.replace("_", " ").replace("70pct", "70%")
    s = re.sub(r"\bpbs\b", "PBS", s)
    s = re.sub(r"\bdna\b", "DNA", s)
    s = re.sub(r"\bpcr\b", "PCR", s)
    return s


def parse_trial_id(fname: str) -> tuple[str, int]:
    stem = Path(fname).stem
    m = re.match(r"P\d+_(\d+)_\d+", stem)
    return stem, int(m.group(1)) if m else -1


def read_segments(ann_path: Path) -> list[tuple[float, float, str]]:
    segs = []
    with open(ann_path, newline="") as f:
        for row in csv.DictReader(f):
            task = (row.get("task") or "").strip()
            if not task:
                continue
            try:
                segs.append((float(row["start_sec"]), float(row["end_sec"]), task))
            except (KeyError, ValueError):
                continue
    segs.sort(key=lambda x: x[0])
    return segs


def step_spans(segs: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for s, e, t in segs:
        if out and out[-1][2] == t:
            out[-1] = (out[-1][0], e, t)
        else:
            out.append((s, e, t))
    return out


def video_meta(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def resolve_video(tid: str, videos_dir: str, mistake_dir: str) -> str | None:
    for p in (
        Path(videos_dir) / f"{tid}.mp4",
        Path(mistake_dir) / f"{tid}.mp4",
        Path(videos_dir) / "finebio_videos_w640" / f"{tid}.mp4",
    ):
        if p.exists():
            return str(p)
    return None


def link_video(src: str, videos_out: Path, tid: str) -> str:
    videos_out.mkdir(parents=True, exist_ok=True)
    dst = videos_out / f"{tid}.mp4"
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(os.path.abspath(src))
    return f"videos/{tid}.mp4"


def sample_indices_in_span(s: float, e: float, fps: float, nframes: int, k: int) -> list[int]:
    if e <= s:
        e = s + 1.0 / max(fps, 1.0)
    times = [s + i * (e - s) / max(k - 1, 1) for i in range(k)]
    return [min(max(int(round(t * fps)), 0), nframes - 1) for t in times]


def build_chunks_from_spans(
    spans: list[tuple[float, float, str]],
    fps: float,
    nframes: int,
    frames_per_chunk: int,
    steps_per_chunk: int,
) -> list[dict]:
    """Group consecutive steps into chunks."""
    chunks = []
    i = 0
    while i < len(spans):
        group = spans[i : i + steps_per_chunk]
        t0, t1 = group[0][0], group[-1][1]
        idxs = sample_indices_in_span(t0, t1, fps, nframes, frames_per_chunk)
        chunks.append({
            "chunk_id": len(chunks),
            "t0": t0,
            "t1": t1,
            "steps": [t for _, _, t in group],
            "expected_step": group[0][2],
            "frame_indices": idxs,
        })
        i += steps_per_chunk
    return chunks


def corrupt_drop(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str]:
    """Drop one interior step. Halt at the chunk that first misses it."""
    if len(spans) < 3:
        return spans, -1, "", ""
    drop_i = random.randint(1, len(spans) - 2)
    dropped = spans[drop_i][2]
    new_spans = spans[:drop_i] + spans[drop_i + 1 :]
    # Halt when we reach the index where dropped step should have appeared
    halt_step_idx = drop_i  # first remaining span after drop position
    reason = f"Missing step: '{humanize(dropped)}' was not observed."
    return new_spans, halt_step_idx, "missing_step", reason


def corrupt_insert(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str]:
    if len(spans) < 3:
        return spans, -1, "", ""
    ins_i = random.randint(1, len(spans) - 2)
    dup = spans[ins_i]
    new_spans = spans[: ins_i + 1] + [dup] + spans[ins_i + 1 :]
    halt_step_idx = ins_i + 1  # the duplicated occurrence
    reason = f"Redundant step: '{humanize(dup[2])}' is performed twice."
    return new_spans, halt_step_idx, "redundant_step", reason


def corrupt_shuffle(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str]:
    if len(spans) < 3:
        return spans, -1, "", ""
    new_spans = spans.copy()
    # Swap two interior steps
    i, j = sorted(random.sample(range(1, len(spans) - 1), 2))
    new_spans[i], new_spans[j] = new_spans[j], new_spans[i]
    halt_step_idx = i
    reason = (
        f"Wrong order: expected '{humanize(spans[i][2])}' but observed "
        f"'{humanize(new_spans[i][2])}'."
    )
    return new_spans, halt_step_idx, "wrong_order", reason


def emit_prefix_samples(
    tid: str,
    rel_video: str,
    proto: int,
    pid: int,
    chunks: list[dict],
    halt_chunk_id: int,
    error_type: str,
    reason: str,
    integrity: int,
    corruption: str | None,
) -> list[dict]:
    """Emit one training sample per prefix ending at chunk k (stop after first HALT)."""
    samples = []
    max_k = halt_chunk_id if halt_chunk_id >= 0 else len(chunks) - 1
    for k in range(max_k + 1):
        is_halt = halt_chunk_id >= 0 and k == halt_chunk_id
        label = error_type if is_halt else "continue"
        prefix = chunks[: k + 1]
        # Use last chunk frames as visual input; keep prefix metadata for FSM
        last = prefix[-1]
        step_vocab_idx = -100  # filled later if we have a global step map
        samples.append({
            "id": f"{tid}_{corruption or 'intact'}_prefix{k}",
            "video": rel_video,
            "protocol_id": pid,
            "protocol": proto,
            "protocol_name": PROTOCOL_NAMES[proto],
            "integrity": integrity,
            "corruption": corruption,
            "prefix_len": k + 1,
            "frame_indices": last["frame_indices"],
            "t0": last["t0"],
            "t1": last["t1"],
            "expected_step": last["expected_step"],
            "halt_label": HALT_LABELS.get(label, 0),
            "halt_name": label,
            "reason": reason if is_halt else "",
            "conversations": build_conv(proto, label, reason if is_halt else ""),
            "prefix_steps": [c["expected_step"] for c in prefix],
        })
    return samples


def build_conv(proto: int, label: str, reason: str) -> list[dict]:
    q = (
        "<image>\nYou are monitoring a wet-lab experiment in real time. "
        f"The intended protocol is protocol {proto}: {PROTOCOL_NAMES[proto]}. "
        "Based on the latest video chunk (and prior context), decide whether to "
        "CONTINUE or HALT. If HALT, state the error type and brief reason."
    )
    if label == "continue":
        a = "CONTINUE. Observed steps still match the protocol."
    else:
        a = f"HALT. error_type={label}. {reason}"
    return [{"from": "human", "value": q}, {"from": "gpt", "value": a}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
    ap.add_argument("--ann-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations")
    ap.add_argument("--mistake-videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/FineBioStreaming/data/streaming_v1")
    ap.add_argument("--frames-per-chunk", type=int, default=8)
    ap.add_argument("--steps-per-chunk", type=int, default=2)
    ap.add_argument("--synth-per-trial", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    videos_out = out / "videos"
    out.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(Path(args.ann_dir).glob("P*.txt"))
    if args.limit:
        ann_files = ann_files[: args.limit]

    all_segs = {}
    for af in ann_files:
        tid, _ = parse_trial_id(af.name)
        all_segs[tid] = read_segments(af)

    # Global step vocabulary for L_step
    step_counter: Counter = Counter()
    for segs in all_segs.values():
        for _, _, t in step_spans(segs):
            step_counter[t] += 1
    step_vocab = {s: i for i, (s, _) in enumerate(step_counter.most_common())}
    (out / "step_vocab.json").write_text(json.dumps(step_vocab, indent=2))
    (out / "halt_labels.json").write_text(json.dumps(HALT_LABELS, indent=2))
    (out / "protocol_reference.json").write_text(json.dumps(
        {str(p): PROTOCOL_NAMES[p] for p in PROTOCOL_IDS}, indent=2))

    samples: list[dict] = []
    counts = Counter()
    skipped = []
    corruptors = [
        ("drop", corrupt_drop),
        ("insert", corrupt_insert),
        ("shuffle", corrupt_shuffle),
    ]

    for tid, segs in all_segs.items():
        _, proto = parse_trial_id(tid)
        if proto not in PROTOCOL_NAMES or not segs:
            continue
        video = resolve_video(tid, args.videos_dir, args.mistake_videos_dir)
        if video is None:
            skipped.append(tid)
            continue
        fps, nframes = video_meta(video)
        if nframes <= 0:
            skipped.append(tid)
            continue
        spans = step_spans(segs)
        if len(spans) < 3:
            skipped.append(tid)
            continue

        rel = link_video(video, videos_out, tid)
        pid = PROTO_TO_CLASS[proto]
        is_real = tid in REAL_MISTAKES

        # Intact: all CONTINUE prefixes
        intact_chunks = build_chunks_from_spans(
            spans, fps, nframes, args.frames_per_chunk, args.steps_per_chunk)
        for s in emit_prefix_samples(
            tid, rel, proto, pid, intact_chunks, -1, "continue", "",
            integrity=0 if is_real else 1,
            corruption=None if not is_real else "real_mistake",
        ):
            s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
            # Real mistakes: force HALT on last chunk with FineBio reason
            if is_real and s["prefix_len"] == len(intact_chunks):
                et, reason = REAL_MISTAKES[tid]
                s["halt_label"] = HALT_LABELS.get(et, 1)
                s["halt_name"] = et
                s["reason"] = reason
                s["integrity"] = 0
                s["conversations"] = build_conv(proto, et, reason)
                counts[f"real_{et}"] += 1
            else:
                counts["continue"] += 1
            samples.append(s)

        if is_real:
            continue

        # Synthetic corruptions
        for k in range(args.synth_per_trial):
            mode, fn = corruptors[k % len(corruptors)]
            new_spans, halt_step_idx, et, reason = fn(spans)
            if halt_step_idx < 0 or not et:
                continue
            chunks = build_chunks_from_spans(
                new_spans, fps, nframes, args.frames_per_chunk, args.steps_per_chunk)
            # Map step index → chunk id
            halt_chunk = min(halt_step_idx // args.steps_per_chunk, len(chunks) - 1)
            for s in emit_prefix_samples(
                tid, rel, proto, pid, chunks, halt_chunk, et, reason,
                integrity=0, corruption=mode,
            ):
                s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
                counts[et if s["halt_name"] != "continue" else "continue"] += 1
                counts[f"synth_{mode}"] += 1
                samples.append(s)

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    meta = {
        "task": "finebio_streaming_halt",
        "frames_per_chunk": args.frames_per_chunk,
        "steps_per_chunk": args.steps_per_chunk,
        "synth_per_trial": args.synth_per_trial,
        "n_step_vocab": len(step_vocab),
        "halt_labels": HALT_LABELS,
        "architecture": "two_stage_halt_step_reason",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps(val))

    print(f"[done] trials={len(all_segs)} skipped={len(skipped)}")
    print(f"  samples={len(samples)} train={len(train)} val={len(val)}")
    print(f"  counts={dict(counts)}")
    print(f"  step_vocab={len(step_vocab)} -> {out}")


if __name__ == "__main__":
    main()
