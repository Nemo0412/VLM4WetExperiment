#!/usr/bin/env python3
"""Build a LLaVA-NeXT-Video instruction-tuning dataset from FineBio.

Unlike the LLaVA-1.5 grid pipeline, samples reference raw mp4 paths. Frame
sampling happens at train/inference time (default: uniform 32 frames). Synthetic
corruption samples store explicit ``frame_indices`` so dropped/shuffled steps
still change which moments are shown.

Output layout (under --out-dir):
  videos/              symlinks to source mp4 files
  train.json / val.json
  protocol_reference.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_finebio_llava import (  # noqa: E402
    PROTOCOL_IDS,
    PROTOCOL_NAMES,
    PROTO_TO_CLASS,
    REAL_MISTAKES,
    canonical_protocols,
    collapse_steps,
    compliance_conv,
    humanize,
    parse_trial_id,
    read_segments,
    scene_conv,
    times_drop_step,
    times_shuffle,
    times_uniform,
    video_meta,
)


def resolve_video(tid: str, videos_dir: str, mistake_dir: str) -> str | None:
    p1 = Path(videos_dir) / f"{tid}.mp4"
    if p1.exists():
        return str(p1)
    p2 = Path(mistake_dir) / f"{tid}.mp4"
    if p2.exists():
        return str(p2)
    nested = Path(videos_dir) / "finebio_videos_w640" / f"{tid}.mp4"
    if nested.exists():
        return str(nested)
    return None


def link_video(src: str, videos_dir: Path, tid: str) -> str:
    videos_dir.mkdir(parents=True, exist_ok=True)
    dst = videos_dir / f"{tid}.mp4"
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(os.path.abspath(src))
    return f"videos/{tid}.mp4"


def to_idx(times: list[float], fps: float, nframes: int) -> list[int]:
    return [min(int(round(t * fps)), nframes - 1) for t in times]


def scene_conv_video(proto: int, steps: list[str]) -> list[dict]:
    conv = scene_conv(proto, steps)
    conv[0]["value"] = conv[0]["value"].replace(
        "These frames are sampled in temporal order (left-to-right, top-to-bottom) from a first-person video",
        "These frames are uniformly sampled in temporal order from a first-person video",
    )
    return conv


def compliance_conv_video(proto: int, followed: bool, reason: str = "") -> list[dict]:
    conv = compliance_conv(proto, followed, reason)
    conv[0]["value"] = conv[0]["value"].replace(
        "These frames are sampled in temporal order from a first-person",
        "These frames are uniformly sampled in temporal order from a first-person",
    )
    return conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
    ap.add_argument("--mistake-videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")
    ap.add_argument("--ann-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/Llava/data/finebio_video")
    ap.add_argument("--num-frames", type=int, default=32,
                    help="reference frame count for uniform sampling (stored in metadata)")
    ap.add_argument("--synth-per-trial", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    videos_out = out / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)

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

    samples: list[dict] = []
    n_scene = n_ok = n_realbad = n_synbad = 0
    skipped = []

    for tid, segs in all_segs.items():
        _, proto = parse_trial_id(tid)
        if proto not in PROTOCOL_NAMES or not segs:
            continue
        video = resolve_video(tid, args.videos_dir, args.mistake_videos_dir)
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

        rel_video = link_video(video, videos_out, tid)
        intact_idx = to_idx(times_uniform(dur, args.num_frames), fps, nframes)

        if not is_real_mistake:
            samples.append({
                "id": f"{tid}_intact_scene",
                "video": rel_video,
                "protocol_id": pid,
                "integrity": 1,
                "conversations": scene_conv_video(proto, steps),
            })
            n_scene += 1

        if is_real_mistake:
            samples.append({
                "id": f"{tid}_intact_comp",
                "video": rel_video,
                "protocol_id": pid,
                "integrity": 0,
                "conversations": compliance_conv_video(proto, False, REAL_MISTAKES[tid]),
            })
            n_realbad += 1
        else:
            samples.append({
                "id": f"{tid}_intact_comp",
                "video": rel_video,
                "protocol_id": pid,
                "integrity": 1,
                "conversations": compliance_conv_video(proto, True),
            })
            n_ok += 1

        if not is_real_mistake:
            for k in range(args.synth_per_trial):
                if k % 2 == 0:
                    times, dropped = times_drop_step(segs, args.num_frames)
                    reason = (
                        f"One or more steps appear to be missing "
                        f"(e.g. '{humanize(dropped)}' is not observed)."
                    ) if dropped else ""
                else:
                    times = times_shuffle(segs, args.num_frames)
                    reason = "The steps appear to be performed in the wrong order."
                if not times or not reason:
                    continue
                idx = to_idx(times, fps, nframes)
                samples.append({
                    "id": f"{tid}_synth{k}_comp",
                    "video": rel_video,
                    "frame_indices": idx,
                    "protocol_id": pid,
                    "integrity": 0,
                    "conversations": compliance_conv_video(proto, False, reason),
                })
                n_synbad += 1

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    meta = {"num_frames_default": args.num_frames, "format": "llava-next-video"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "train.json").write_text(json.dumps(train, indent=2))
    (out / "val.json").write_text(json.dumps(val, indent=2))

    print(f"[done] trials={len(all_segs)} skipped={len(skipped)}")
    if skipped:
        print(f"  skipped ids: {skipped[:20]}{'...' if len(skipped) > 20 else ''}")
    print(f"  scene={n_scene} compliant_ok={n_ok} real_mistake={n_realbad} synth_mistake={n_synbad}")
    print(f"  total samples={len(samples)} -> train={len(train)} val={len(val)}")
    print(f"  videos -> {videos_out}")


if __name__ == "__main__":
    main()
