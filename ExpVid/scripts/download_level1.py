#!/usr/bin/env python3
"""Resume ExpVid level_1 download with 429 backoff. No GPU needed."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def count_mp4(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mp4"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/ExpVid")
    ap.add_argument("--target", type=int, default=3463, help="expected level_1 mp4 count")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out_dir)
    videos = out / "videos" / "level_1"
    videos.mkdir(parents=True, exist_ok=True)

    # Use explicit token if set; else anonymous (ExpVid is public).
    # Do NOT fall back to token=True — a stale HF_HOME token breaks downloads.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False

    round_i = 0
    while round_i < args.max_rounds:
        round_i += 1
        n = count_mp4(videos)
        print(f"[round {round_i}] local_mp4={n}/{args.target}", flush=True)
        if n >= args.target:
            print("[DONE] level_1 complete", flush=True)
            return

        try:
            snapshot_download(
                repo_id="OpenGVLab/ExpVid",
                repo_type="dataset",
                local_dir=str(out),
                allow_patterns=["videos/level_1/**", "annotations/**", "README.md"],
                max_workers=args.max_workers,
                token=token,
            )
        except Exception as e:
            msg = str(e)
            print(f"[WARN] download error: {type(e).__name__}: {msg[:400]}", flush=True)
            # exponential-ish backoff for rate limits
            wait = 90
            if "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower():
                wait = min(900, 120 * round_i)  # 2m, 4m, 6m... up to 15m
            print(f"[INFO] sleep {wait}s then resume", flush=True)
            time.sleep(wait)
            continue

        n2 = count_mp4(videos)
        print(f"[INFO] after attempt local_mp4={n2}", flush=True)
        if n2 >= args.target:
            print("[DONE] level_1 complete", flush=True)
            return
        if n2 <= n:
            wait = min(600, 45 * round_i)
            print(f"[INFO] no progress; sleep {wait}s", flush=True)
            time.sleep(wait)

    n = count_mp4(videos)
    raise SystemExit(f"[FAIL] stopped with local_mp4={n}/{args.target}")


if __name__ == "__main__":
    main()
