"""Disk frame cache: video_stem + frame_indices → uint8 (T,H,W,C) .npy."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


def cache_key(video_rel: str, indices: list[int]) -> str:
    stem = Path(video_rel).stem
    idx = "_".join(str(i) for i in indices)
    raw = f"{stem}|{idx}"
    if len(raw) > 180:
        return f"{stem}_{hashlib.md5(idx.encode()).hexdigest()[:16]}"
    return f"{stem}__{idx}"


def cache_path(cache_dir: Path | str, video_rel: str, indices: list[int]) -> Path:
    return Path(cache_dir) / f"{cache_key(video_rel, indices)}.npy"


class FrameCacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.decode_s = 0.0
        self.load_s = 0.0
        self._lock = threading.Lock()

    def record(self, hit: bool, dt: float):
        with self._lock:
            if hit:
                self.hits += 1
                self.load_s += dt
            else:
                self.misses += 1
                self.decode_s += dt

    def snapshot(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "decode_s": round(self.decode_s, 3),
                "load_s": round(self.load_s, 3),
                "avg_decode_ms": round(1000 * self.decode_s / max(self.misses, 1), 1),
                "avg_load_ms": round(1000 * self.load_s / max(self.hits, 1), 1),
            }


def load_frames_cached(
    video_path: str,
    indices: list[int],
    cache_dir: Path | str | None,
    video_rel: str | None = None,
    stats: FrameCacheStats | None = None,
    write_on_miss: bool = True,
) -> tuple[np.ndarray, bool]:
    """Load frames from cache if present, else decord (+ optional write-through).

    Returns (frames, hit).
    """
    rel = video_rel or Path(video_path).name
    if cache_dir is not None:
        path = cache_path(cache_dir, rel, indices)
        if path.exists() and path.stat().st_size > 0:
            t0 = time.perf_counter()
            frames = np.load(path)
            if stats is not None:
                stats.record(True, time.perf_counter() - t0)
            return frames, True

    t0 = time.perf_counter()
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(i, 0), n - 1) for i in indices]
    frames = vr.get_batch(idxs).asnumpy()
    dt = time.perf_counter() - t0
    if stats is not None:
        stats.record(False, dt)

    if cache_dir is not None and write_on_miss:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        out = cache_path(cache_dir, rel, indices)
        tmp = out.with_suffix(".tmp.npy")
        try:
            np.save(tmp, frames)
            os.replace(tmp, out)
        except OSError:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return frames, False
