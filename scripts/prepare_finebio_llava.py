#!/usr/bin/env python3
"""Build a LLaVA instruction-tuning dataset from FineBio.

Two capabilities are targeted:
  1. Scene / protocol recognition  -> "which protocol is this and what are the steps"
  2. Mistake detection             -> "does it follow the protocol; if not, what is wrong"

Because FineBio only has 11 human-labeled mistake trials, we additionally
synthesize corrupted trials (drop a step segment / shuffle step order) from the
correct trials. Every emitted sample carries two extra fields used by the
custom-loss trainer:

  - protocol_id : int in [0, 6]           (for the auxiliary protocol head, L_proto)
  - integrity   : 1 = intact, 0 = corrupted (for the self-supervised order head, L_order)

Standard LLaVA training ignores those extra fields; our custom trainer reads them.

Output layout (under --out-dir):
  images/<sample_id>.jpg
  train.json
  val.json
  protocol_reference.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from PIL import Image

# ----------------------------------------------------------------------------
# Protocol metadata (names inferred from step sequences; FineBio ships no text)
# ----------------------------------------------------------------------------
PROTOCOL_NAMES = {
    1: "Cell lysate collection (single PBS wash)",
    2: "Cell lysate collection (double PBS wash)",
    3: "Magnetic-bead DNA extraction (single ethanol wash)",
    4: "Magnetic-bead DNA extraction (double ethanol wash)",
    5: "PCR reaction setup with 8-tube strips",
    6: "Spin-column DNA extraction (two wash steps)",
    7: "Spin-column DNA extraction (three wash steps)",
}
PROTOCOL_IDS = sorted(PROTOCOL_NAMES.keys())  # 1..7
PROTO_TO_CLASS = {p: i for i, p in enumerate(PROTOCOL_IDS)}  # 1->0 ... 7->6

# Human-readable mistake descriptions (FineBio paper Table 13).
REAL_MISTAKES = {
    "P06_03_02": "One or more steps are missing: the sterile water wash is skipped (about 6 steps missing).",
    "P11_06_01": "There are redundant steps: an extra wash buffer step is performed (about 3 redundant steps).",
    "P17_02_02": "There are redundant steps: an extra PBS wash is performed (about 6 redundant steps).",
    "P03_03_02": "A within-step operation is missing: no pipetting during suction.",
    "P03_04_02": "A within-step operation is missing: no pipetting during suction.",
    "P07_04_01": "A within-step operation is missing: the vortex step is skipped.",
    "P17_07_01": "Steps are out of order: 'dispense solution' and 'detach spin column and insert to new tube' are swapped.",
    "P18_02_01": "A within-step operation is missing: the spindown step is skipped.",
    "P18_05_01": "The last step is missing (due to missing frames).",
    "P24_07_01": "A step is forgotten: dispensing the solution after the second-to-last spindown is skipped.",
    "P28_06_02": "Steps are out of order: 'dispense solution' and 'detach spin column and insert to new tube' are swapped.",
}
MAJOR_MISTAKES = {"P06_03_02", "P11_06_01", "P17_02_02"}


def humanize(task: str) -> str:
    """add_70pct_ethanol -> add 70% ethanol; add_pbs -> add PBS."""
    s = task.replace("_", " ")
    s = s.replace("70pct", "70%")
    s = re.sub(r"\bpbs\b", "PBS", s)
    s = re.sub(r"\bdna\b", "DNA", s)
    s = re.sub(r"\bpcr\b", "PCR", s)
    s = re.sub(r"\bpcrmix\b", "PCR mix", s)
    return s


def parse_trial_id(fname: str) -> tuple[str, int]:
    """P06_03_01.txt -> ('P06_03_01', 3)  (protocol = middle field)."""
    stem = Path(fname).stem
    m = re.match(r"P\d+_(\d+)_\d+", stem)
    proto = int(m.group(1)) if m else -1
    return stem, proto


def read_segments(ann_path: Path) -> list[tuple[float, float, str]]:
    """Return step-level segments [(start, end, task), ...] in temporal order."""
    segs: list[tuple[float, float, str]] = []
    with open(ann_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = (row.get("task") or "").strip()
            if not task:
                continue  # atomic-operation rows have empty task
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, ValueError):
                continue
            segs.append((start, end, task))
    segs.sort(key=lambda x: x[0])
    return segs


def collapse_steps(segs: list[tuple[float, float, str]]) -> list[str]:
    """Merge consecutive identical tasks into a step list."""
    steps: list[str] = []
    for _, _, task in segs:
        if not steps or steps[-1] != task:
            steps.append(task)
    return steps


def canonical_protocols(all_segs: dict[str, list]) -> dict[int, list[str]]:
    """Pick the most common step-list length's representative per protocol."""
    by_proto: dict[int, list[list[str]]] = defaultdict(list)
    for tid, segs in all_segs.items():
        _, proto = parse_trial_id(tid)
        if proto in PROTOCOL_NAMES:
            by_proto[proto].append(collapse_steps(segs))
    ref: dict[int, list[str]] = {}
    for proto, lists in by_proto.items():
        # representative = the modal step-sequence (as tuple) among correct trials
        counter = Counter(tuple(x) for x in lists)
        ref[proto] = list(counter.most_common(1)[0][0])
    return ref


def video_meta(video: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def decode_frames_by_index(video: str, indices: list[int]) -> dict[int, Image.Image]:
    """Single sequential pass: grab() to skip, retrieve() only at wanted indices.

    Much faster than repeated CAP_PROP_POS_MSEC seeks (which decode from the
    nearest keyframe every time).
    """
    wanted = sorted(set(i for i in indices if i >= 0))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    out: dict[int, Image.Image] = {}
    wi = 0
    cur = -1
    last_frame = None
    max_idx = wanted[-1]
    while wi < len(wanted):
        target = wanted[wi]
        # advance via grab() until we reach target
        while cur < target:
            ok = cap.grab()
            if not ok:
                break
            cur += 1
        if cur < target:
            # ran past end of video; reuse last decoded frame if any
            if last_frame is not None:
                out[target] = last_frame
            wi += 1
            continue
        ok, frame = cap.retrieve()
        if ok:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out[target] = img
            last_frame = img
        wi += 1
        if cur >= max_idx and wi >= len(wanted):
            break
    cap.release()
    return out


def make_grid(frames: list[Image.Image], cell: int = 336) -> Image.Image:
    n = len(frames)
    if n == 0:
        raise RuntimeError("no frames")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    for i, fr in enumerate(frames):
        thumb = fr.copy()
        thumb.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x = (i % cols) * cell + (cell - thumb.width) // 2
        y = (i // cols) * cell + (cell - thumb.height) // 2
        grid.paste(thumb, (x, y))
    return grid


# --- frame-time planners -----------------------------------------------------

def times_uniform(duration: float, num: int) -> list[float]:
    if duration <= 0:
        return []
    return [i * duration / max(num - 1, 1) for i in range(num)]


def times_drop_step(segs, num: int) -> tuple[list[float], str]:
    """Exclude one random step segment, sample uniformly over the rest."""
    steps = collapse_steps(segs)
    if len(steps) < 3:
        return [], ""
    # group segments by contiguous step, choose one interior step to drop
    dropped = random.choice(steps[1:-1]) if len(steps) > 2 else random.choice(steps)
    kept = [(s, e) for (s, e, t) in segs if t != dropped]
    if not kept:
        return [], ""
    spans = [(s, e) for s, e in kept]
    total = sum(e - s for s, e in spans)
    if total <= 0:
        return [], ""
    times = []
    for k in range(num):
        target = k * total / max(num - 1, 1)
        acc = 0.0
        for s, e in spans:
            if acc + (e - s) >= target:
                times.append(s + (target - acc))
                break
            acc += e - s
        else:
            times.append(spans[-1][1])
    return times, dropped


def times_shuffle(segs, num: int) -> list[float]:
    """Reorder step segments, then sample across the reordered timeline."""
    steps = collapse_steps(segs)
    if len(steps) < 3:
        return []
    # build per-step time spans and shuffle their order
    order = list(range(len(segs)))
    random.shuffle(order)
    spans = [(segs[i][0], segs[i][1]) for i in order]
    total = sum(e - s for s, e in spans)
    if total <= 0:
        return []
    times = []
    for k in range(num):
        target = k * total / max(num - 1, 1)
        acc = 0.0
        for s, e in spans:
            if acc + (e - s) >= target:
                times.append(s + (target - acc))
                break
            acc += e - s
        else:
            times.append(spans[-1][1])
    return times


# --- conversation builders ----------------------------------------------------

def steps_to_text(steps: list[str]) -> str:
    return "; ".join(f"{i+1}) {humanize(s)}" for i, s in enumerate(steps))


def scene_conv(proto: int, steps: list[str]) -> list[dict]:
    q = (
        "<image>\nThese frames are sampled in temporal order (left-to-right, "
        "top-to-bottom) from a first-person video of a wet-lab biology experiment. "
        "Identify which experimental protocol is being performed and summarize the "
        "main steps you observe."
    )
    a = (
        f"This is protocol {proto}: {PROTOCOL_NAMES[proto]}. "
        f"The main steps are: {steps_to_text(steps)}."
    )
    return [{"from": "human", "value": q}, {"from": "gpt", "value": a}]


def compliance_conv(proto: int, followed: bool, reason: str = "") -> list[dict]:
    q = (
        "<image>\nThese frames are sampled in temporal order from a first-person "
        f"wet-lab video. The intended protocol is protocol {proto}: "
        f"{PROTOCOL_NAMES[proto]}. Did the experimenter correctly follow this "
        "protocol? Answer 'FOLLOWED' or 'NOT FOLLOWED' and briefly justify."
    )
    if followed:
        a = "FOLLOWED. The observed steps match the reference protocol in the expected order, with no missing, extra, or reordered steps."
    else:
        a = f"NOT FOLLOWED. {reason}"
    return [{"from": "human", "value": q}, {"from": "gpt", "value": a}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
    ap.add_argument("--mistake-videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")
    ap.add_argument("--ann-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/Llava/data/finebio_llava")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--cell", type=int, default=336)
    ap.add_argument("--synth-per-trial", type=int, default=2,
                    help="synthetic corrupted grids per correct trial")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #trials")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = Path(args.ann_dir)
    ann_files = sorted(ann_dir.glob("P*.txt"))
    if args.limit:
        ann_files = ann_files[: args.limit]

    all_segs: dict[str, list] = {}
    for af in ann_files:
        tid, _ = parse_trial_id(af.name)
        all_segs[tid] = read_segments(af)

    ref = canonical_protocols(all_segs)
    (out / "protocol_reference.json").write_text(
        json.dumps(
            {str(p): {"name": PROTOCOL_NAMES[p], "steps": ref.get(p, [])} for p in PROTOCOL_IDS},
            indent=2,
        )
    )

    def resolve_video(tid: str) -> str | None:
        p1 = Path(args.videos_dir) / f"{tid}.mp4"
        if p1.exists():
            return str(p1)
        p2 = Path(args.mistake_videos_dir) / f"{tid}.mp4"
        if p2.exists():
            return str(p2)
        return None

    samples: list[dict] = []
    n_scene = n_ok = n_realbad = n_synbad = 0
    skipped = []

    for tid, segs in all_segs.items():
        _, proto = parse_trial_id(tid)
        if proto not in PROTOCOL_NAMES or not segs:
            continue
        video = resolve_video(tid)
        if video is None:
            skipped.append(tid)
            continue
        steps = collapse_steps(segs)
        pid = PROTO_TO_CLASS[proto]
        is_real_mistake = tid in REAL_MISTAKES
        fps, nframes = video_meta(video)
        dur = (nframes / fps) if fps else 0.0
        if dur <= 0:
            skipped.append(tid)
            continue

        def to_idx(times: list[float]) -> list[int]:
            return [min(int(round(t * fps)), nframes - 1) for t in times]

        # ---- plan every grid for this trial, then decode in ONE pass ----
        plans: list[tuple[str, list[int]]] = []  # (grid_name, frame_indices)
        intact_times = times_uniform(dur, args.num_frames)
        plans.append(("intact", to_idx(intact_times)))

        synth_specs: list[tuple[str, list[int], str]] = []  # (name, idx, reason)
        if not is_real_mistake:
            for k in range(args.synth_per_trial):
                if k % 2 == 0:
                    times, dropped = times_drop_step(segs, args.num_frames)
                    reason = (f"One or more steps appear to be missing "
                              f"(e.g. '{humanize(dropped)}' is not observed).") if dropped else ""
                else:
                    times = times_shuffle(segs, args.num_frames)
                    reason = "The steps appear to be performed in the wrong order."
                if not times or not reason:
                    continue
                idx = to_idx(times)
                synth_specs.append((f"synth{k}", idx, reason))
                plans.append((f"synth{k}", idx))

        union = sorted(set(i for _, idxs in plans for i in idxs))
        decoded = decode_frames_by_index(video, union)
        if not decoded:
            skipped.append(tid)
            continue

        def build(idxs: list[int]) -> list[Image.Image]:
            return [decoded[i] for i in idxs if i in decoded]

        # ---- intact grid (used for scene + compliance) ----
        frames = build(to_idx(intact_times))
        if not frames:
            skipped.append(tid)
            continue
        grid_id = f"{tid}_intact"
        make_grid(frames, args.cell).save(img_dir / f"{grid_id}.jpg", quality=90)

        # scene recognition (skip real-mistake trials to keep step labels clean)
        if not is_real_mistake:
            samples.append({
                "id": f"{grid_id}_scene",
                "image": f"images/{grid_id}.jpg",
                "protocol_id": pid,
                "integrity": 1,
                "conversations": scene_conv(proto, steps),
            })
            n_scene += 1

        # compliance label for the intact grid
        if is_real_mistake:
            samples.append({
                "id": f"{grid_id}_comp",
                "image": f"images/{grid_id}.jpg",
                "protocol_id": pid,
                "integrity": 0,
                "conversations": compliance_conv(proto, False, REAL_MISTAKES[tid]),
            })
            n_realbad += 1
        else:
            samples.append({
                "id": f"{grid_id}_comp",
                "image": f"images/{grid_id}.jpg",
                "protocol_id": pid,
                "integrity": 1,
                "conversations": compliance_conv(proto, True),
            })
            n_ok += 1

        # ---- synthetic corrupted grids (only from correct trials) ----
        for name, idx, reason in synth_specs:
            cframes = build(idx)
            if not cframes:
                continue
            cid = f"{tid}_{name}"
            make_grid(cframes, args.cell).save(img_dir / f"{cid}.jpg", quality=90)
            samples.append({
                "id": f"{cid}_comp",
                "image": f"images/{cid}.jpg",
                "protocol_id": pid,
                "integrity": 0,
                "conversations": compliance_conv(proto, False, reason),
            })
            n_synbad += 1

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    (out / "train.json").write_text(json.dumps(train, indent=2))
    (out / "val.json").write_text(json.dumps(val, indent=2))

    print(f"[done] trials={len(all_segs)} skipped={len(skipped)}")
    if skipped:
        print(f"  skipped ids: {skipped}")
    print(f"  scene={n_scene} compliant_ok={n_ok} real_mistake={n_realbad} synth_mistake={n_synbad}")
    print(f"  total samples={len(samples)} -> train={len(train)} val={len(val)}")
    print(f"  images -> {img_dir}")


if __name__ == "__main__":
    main()
