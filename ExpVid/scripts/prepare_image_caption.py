#!/usr/bin/env python3
"""Prepare ExpVid image–caption pairs by extracting a mid-frame per clip.

Writes:
  frames/{video_id}/{clip}.jpg
  pairs_all.jsonl   — all unique (image, asr_caption, video_path)
  pairs_train.jsonl / pairs_val.jsonl — SSL split by video_id
  qa_level1.jsonl   — level-1 MCQ rows with local image path
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def extract_mid_frame(video_path: Path, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    if not video_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    if len(vr) == 0:
        return False
    idx = len(vr) // 2
    frame = vr[idx].asnumpy()
    Image.fromarray(frame).save(out_path, quality=90)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ll5914/Labos/ExpVid")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl-max-train", type=int, default=2000,
                    help="Cap SSL train pairs (0=all)")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = root / "annotations" / "level1"
    frame_dir = root / "frames"
    out_dir = root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- unique caption pairs from all level1 tasks ---
    by_clip: dict[str, dict] = {}
    qa_rows = []
    for f in sorted(ann_dir.glob("*.jsonl")):
        task = f.stem
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            vp = r["video_path"]
            cap = (r.get("asr_caption") or "").strip()
            if vp not in by_clip and cap:
                by_clip[vp] = {
                    "video_path": vp,
                    "asr_caption": cap,
                    "video_id": r.get("video_id"),
                    "category": r.get("category"),
                }
            qa = {
                "id": r["id"],
                "task": task,
                "video_path": vp,
                "asr_caption": cap,
                "question": r["question"],
                "options": r["options"],
                "answer": r["answer"],
                "video_id": r.get("video_id"),
                "category": r.get("category"),
            }
            qa_rows.append(qa)

    print(f"[prep] unique clips with caption={len(by_clip)} qa_rows={len(qa_rows)}", flush=True)

    pairs = []
    miss = 0
    for i, (vp, meta) in enumerate(sorted(by_clip.items()), 1):
        # videos/level_1/55531/clip_12.mp4 -> frames/level_1/55531/clip_12.jpg
        rel = Path(vp)
        img_rel = Path("frames") / rel.parent.relative_to("videos") / (rel.stem + ".jpg")
        img_abs = root / img_rel
        ok = extract_mid_frame(root / vp, img_abs)
        if not ok:
            miss += 1
            continue
        pairs.append({
            **meta,
            "image_path": str(img_rel),
        })
        if i % 100 == 0 or i == len(by_clip):
            print(f"[frames] {i}/{len(by_clip)} ok={len(pairs)} miss={miss}", flush=True)

    # split by video_id for SSL
    rng = random.Random(args.seed)
    vids = sorted({p["video_id"] for p in pairs})
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * args.val_frac)))
    val_vids = set(vids[:n_val])
    train, val = [], []
    for p in pairs:
        (val if p["video_id"] in val_vids else train).append(p)
    if args.ssl_max_train > 0 and len(train) > args.ssl_max_train:
        rng.shuffle(train)
        train = train[: args.ssl_max_train]

    def write_jsonl(path: Path, rows: list[dict]):
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "pairs_all.jsonl", pairs)
    write_jsonl(out_dir / "pairs_train.jsonl", train)
    write_jsonl(out_dir / "pairs_val.jsonl", val)

    # attach image paths to QA
    clip2img = {p["video_path"]: p["image_path"] for p in pairs}
    qa_out = []
    for r in qa_rows:
        img = clip2img.get(r["video_path"])
        if not img:
            continue
        r = dict(r)
        r["image_path"] = img
        qa_out.append(r)
    write_jsonl(out_dir / "qa_level1.jsonl", qa_out)

    meta = {
        "n_pairs": len(pairs),
        "n_train": len(train),
        "n_val": len(val),
        "n_qa": len(qa_out),
        "miss_videos": miss,
        "val_video_ids": sorted(val_vids),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
