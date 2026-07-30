#!/usr/bin/env python3
"""Prefill uint8 clip cache so training never opens mp4 on the hot path.

Key = video stem + frame indices → .npy of shape (T, H, W, C).
Fixed stored indices only (no anticipation jitter) so keys stay stable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_cache import cache_path  # noqa: E402


def collect_clips(jsonl_paths: list[Path]) -> dict[tuple[str, tuple[int, ...]], None]:
    clips: dict[tuple[str, tuple[int, ...]], None] = {}
    for path in jsonl_paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for seg in rec.get("segments") or []:
                    idxs = tuple(int(i) for i in (seg.get("frame_indices") or [0]))
                    clips[(seg["video"], idxs)] = None
                if "video" in rec and "frame_indices" in rec:
                    idxs = tuple(int(i) for i in rec["frame_indices"])
                    clips[(rec["video"], idxs)] = None
    return clips


def process_one(args_tuple):
    video_rel, indices, video_root, cache_dir = args_tuple
    out = cache_path(Path(cache_dir), video_rel, list(indices))
    if out.exists() and out.stat().st_size > 0:
        return video_rel, "skip", 0
    path = os.path.join(video_root, video_rel)
    vr = VideoReader(path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(i, 0), n - 1) for i in indices]
    frames = vr.get_batch(idxs).asnumpy()
    # atomic write
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, frames)
    os.replace(tmp, out)
    return video_rel, "ok", int(frames.nbytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default=None,
                    help="Default: <data-dir>/frame_cache")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "frame_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jsonls = [data_dir / f"{s}.jsonl" for s in args.splits.split(",") if s.strip()]
    clips = collect_clips(jsonls)
    items = list(clips.keys())
    if args.limit:
        items = items[: args.limit]

    jobs = [(v, idxs, str(data_dir), str(cache_dir)) for v, idxs in items]
    done = skip = fail = 0
    bytes_sum = 0
    print(f"[cache] unique_clips={len(jobs)} → {cache_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _id, st, nbytes = fut.result()
                if st == "skip":
                    skip += 1
                else:
                    done += 1
                    bytes_sum += nbytes
            except Exception as e:
                fail += 1
                print(f"[fail] {e}", flush=True)
            if i % 100 == 0 or i == len(futs):
                print(
                    f"[{i}/{len(futs)}] ok={done} skip={skip} fail={fail} "
                    f"new_GB={bytes_sum / 1e9:.2f}",
                    flush=True,
                )

    meta = {
        "n_clips": done + skip,
        "ok": done,
        "skip": skip,
        "fail": fail,
        "dtype": "uint8",
        "layout": "THWC",
        "key": "video_stem__frame_indices",
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] cache={cache_dir} ok={done} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
