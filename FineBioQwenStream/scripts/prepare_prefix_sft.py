#!/usr/bin/env python3
"""Build *protocol-level* prefix SFT data (no steps).

Each P_i = one full FineBio trial video of protocol type i.
Intended plan default: [1,2,3,4,5,6,7] (or a prefix of it).

Positive: P1, P1+P2, ..., P1+...+Pk → CONTINUE
Negative: P2, P1+P3, P1+P2+P4, ... → HALT at first ≤5 frames of the bad protocol video.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol_prompt import (  # noqa: E402
    PROTOCOL_NAMES,
    build_messages,
    type_id,
)

PROTOCOL_IDS = sorted(PROTOCOL_NAMES.keys())


def parse_trial(fname: str) -> tuple[str, int]:
    stem = Path(fname).stem
    m = re.match(r"P\d+_(\d+)_\d+", stem)
    return stem, int(m.group(1)) if m else -1


def video_meta(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def resolve_video(tid: str, videos_dir: str, mistake_dir: str) -> str | None:
    for p in (Path(videos_dir) / f"{tid}.mp4", Path(mistake_dir) / f"{tid}.mp4"):
        if p.exists():
            return str(p)
    return None


def link_video(src: str, videos_out: Path, tid: str) -> str:
    videos_out.mkdir(parents=True, exist_ok=True)
    dst = videos_out / f"{tid}.mp4"
    if not dst.exists() and not dst.is_symlink():
        try:
            dst.symlink_to(os.path.abspath(src))
        except OSError:
            pass
    return f"videos/{tid}.mp4"


def sample_uniform(nframes: int, k: int) -> list[int]:
    if nframes <= 0:
        return [0]
    if nframes == 1:
        return [0] * k
    return [int(round(i * (nframes - 1) / max(k - 1, 1))) for i in range(k)]


def sample_halt_head(nframes: int, halt_frames: int) -> list[int]:
    """First *consecutive* k frames of the bad protocol video (k in 1..5)."""
    h = max(1, min(5, halt_frames, max(nframes, 1)))
    return list(range(h))


def horizon_loss_weight(k: int) -> float:
    """Early preferred; frame-5 is a hard must (high floor weight).

    Preference: 1 best > 2-4 ok > 5 must-detect.
      w(k) = w_early / k  +  w_must * 1[k==5]
    """
    w_early = 1.5
    w_must = 1.0  # ensures k=5 is never under-weighted
    return w_early / float(k) + (w_must if k == 5 else 0.0)


def make_record(
    *,
    sample_id: str,
    intended: list[int],
    segments: list[dict],
    observed: list[int],
    label: str,
    error_type: str,
    reason: str,
    kind: str,
    halt_horizon: int | None = None,
    loss_weight: float = 1.0,
) -> dict:
    messages = build_messages(
        intended, n_segments_seen=len(segments),
        label=label, error_type=error_type, reason=reason,
    )
    return {
        "id": sample_id,
        "intended_protocols": intended,
        "observed_protocols": observed,
        "segments": segments,  # list[{video, frame_indices, protocol_id, trial_id}]
        "label": label,  # continue | halt
        "error_type": error_type or None,
        "type_id": type_id(label, error_type or None),
        "halt_label": 0 if label == "continue" else 1,
        "halt_horizon": halt_horizon,  # 1..5 for HALT; None for CONTINUE
        "loss_weight": float(loss_weight),
        "kind": kind,
        "messages": messages,
    }


def expand_halt_horizons(rec: dict, max_h: int = 5) -> list[dict]:
    """One logical error → 5 supervised views (k=1..5 frames of bad segment)."""
    assert rec["label"] == "halt"
    segs = rec["segments"]
    assert segs and segs[-1].get("is_halt_window"), "last segment must be halt window"
    bad = segs[-1]
    nframes = int(bad.get("nframes") or max(bad["frame_indices"] + [0]) + 1)
    # recover reason text from assistant reply: "HALT. error_type=X. <reason>"
    asst = rec["messages"][1]["content"]
    reason = asst.split(". ", 2)[2] if asst.count(". ") >= 2 else asst
    out = []
    for k in range(1, max_h + 1):
        new_segs = [dict(s) for s in segs[:-1]]
        new_bad = dict(bad)
        new_bad["frame_indices"] = sample_halt_head(nframes, k)
        new_bad["is_halt_window"] = True
        new_segs.append(new_bad)
        w = horizon_loss_weight(k)
        out.append(make_record(
            sample_id=f"{rec['id']}_h{k}",
            intended=rec["intended_protocols"],
            segments=new_segs,
            observed=rec["observed_protocols"],
            label="halt",
            error_type=rec["error_type"] or "",
            reason=reason,
            kind=f"{rec['kind']}_h{k}",
            halt_horizon=k,
            loss_weight=w,
        ))
    return out


def split_pool(pool: dict[int, list[dict]], val_frac: float, test_frac: float, seed: int):
    """Split trial videos per protocol so concat partners don't leak across splits."""
    rng = random.Random(seed)
    train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)
    for pid, items in pool.items():
        ids = items[:]
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_frac))) if n >= 5 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
        if n_test + n_val >= n:
            n_test = max(0, n // 5)
            n_val = max(0, n // 5)
        test[pid] = ids[:n_test]
        val[pid] = ids[n_test: n_test + n_val]
        train[pid] = ids[n_test + n_val:]
    return train, val, test


def pick(pool: dict[int, list[dict]], pid: int, rng: random.Random) -> dict:
    return rng.choice(pool[pid])


def pick_wrong_experiment(
    pool: dict[int, list[dict]],
    intended: list[int],
    expected_next: int | None,
    rng: random.Random,
) -> dict:
    """Pick a *ridiculous* wrong segment: another protocol / experiment video.

    Prefer protocols far from the expected next (and not equal to it).
    """
    candidates = [p for p in pool if pool[p]]
    # strongly avoid the correct next protocol
    avoid = {expected_next} if expected_next is not None else set()
    far = [p for p in candidates if p not in avoid]
    if not far:
        far = candidates
    # prefer ids with large |p - expected|
    if expected_next is not None and len(far) > 1:
        far = sorted(far, key=lambda p: -abs(p - expected_next))
        # sample among the farthest half
        far = far[: max(1, len(far) // 2)]
    pid = rng.choice(far)
    return pick(pool, pid, rng)


def build_split(
    pool: dict[int, list[dict]],
    intended: list[int],
    frames_per_proto: int,
    halt_frames: int,
    halt_oversample: int,
    max_neg: int,
    seed: int,
) -> tuple[list[dict], Counter]:
    rng = random.Random(seed)
    samples: list[dict] = []
    counts = Counter()
    if any(pid not in pool or not pool[pid] for pid in intended):
        return samples, counts

    # ---- positives: P1, P1+P2, ... ----
    for k in range(1, len(intended) + 1):
        segs = []
        observed = []
        for pid in intended[:k]:
            tr = pick(pool, pid, rng)
            segs.append({
                "video": tr["video"],
                "trial_id": tr["tid"],
                "protocol_id": pid,
                "frame_indices": sample_uniform(tr["nframes"], frames_per_proto),
            })
            observed.append(pid)
        samples.append(make_record(
            sample_id=f"pos_k{k}_" + "_".join(str(p) for p in observed),
            intended=intended, segments=segs, observed=observed,
            label="continue", error_type="", reason="", kind=f"pos_prefix{k}",
        ))
        counts["continue"] += 1

    # ---- negatives (exactly 2 mistake types) ----
    negs = []

    # (1) missing_protocol: skip one or more intended protocols
    #     - start at Pj (j>=2): missing P1..P{j-1}
    #     - P1..Pi then jump to Pj (j>i+1): missing P{i+2}..P{j}
    for j in range(1, len(intended)):
        pid = intended[j]
        tr = pick(pool, pid, rng)
        missing = intended[:j]
        segs = [{
            "video": tr["video"],
            "trial_id": tr["tid"],
            "protocol_id": pid,
            "frame_indices": sample_halt_head(tr["nframes"], halt_frames),
            "is_halt_window": True,
            "nframes": tr["nframes"],
        }]
        negs.append(make_record(
            sample_id=f"neg_miss_start_P{j+1}",
            intended=intended, segments=segs, observed=[pid],
            label="halt", error_type="missing_protocol",
            reason=(
                f"Missing protocol(s) "
                + ", ".join(f"P{t+1}(protocol {intended[t]})" for t in range(j))
                + f"; stream jumped to P{j+1}."
            ),
            kind=f"neg_miss_start_{j+1}",
        ))

    for i in range(0, len(intended) - 2):
        for j in range(i + 2, len(intended)):
            segs = []
            observed = []
            for pid in intended[: i + 1]:
                tr = pick(pool, pid, rng)
                segs.append({
                    "video": tr["video"],
                    "trial_id": tr["tid"],
                    "protocol_id": pid,
                    "frame_indices": sample_uniform(tr["nframes"], frames_per_proto),
                })
                observed.append(pid)
            bad_pid = intended[j]
            tr = pick(pool, bad_pid, rng)
            segs.append({
                "video": tr["video"],
                "trial_id": tr["tid"],
                "protocol_id": bad_pid,
                "frame_indices": sample_halt_head(tr["nframes"], halt_frames),
                "is_halt_window": True,
                "nframes": tr["nframes"],
            })
            observed.append(bad_pid)
            missed = intended[i + 1: j]
            negs.append(make_record(
                sample_id=f"neg_miss_{i+1}_{j+1}",
                intended=intended, segments=segs, observed=observed,
                label="halt", error_type="missing_protocol",
                reason=(
                    f"Missing protocol(s) "
                    + ", ".join(
                        f"P{i+2+t}(protocol {m})" for t, m in enumerate(missed)
                    )
                    + f"; after P{i+1} jumped to P{j+1} (protocol {bad_pid})."
                ),
                kind=f"neg_miss_{i+1}_{j+1}",
            ))

    # (2) wrong_execution: current segment is a botched / wrong experiment.
    #     After correct prefix P1..Pi, insert a *wild* video from another experiment
    #     (protocol id ≠ expected next; prefer far-away protocol).
    for i in range(0, len(intended)):
        # i = last correct index; expected next is intended[i] if i==0 empty prefix? 
        # prefix length = i means segments intended[0:i], expected next = intended[i]
        # i from 0..len-1: empty prefix + wrong, or P1..Pi + wrong
        segs = []
        observed = []
        for pid in intended[:i]:
            tr = pick(pool, pid, rng)
            segs.append({
                "video": tr["video"],
                "trial_id": tr["tid"],
                "protocol_id": pid,
                "frame_indices": sample_uniform(tr["nframes"], frames_per_proto),
            })
            observed.append(pid)
        expected_next = intended[i]
        wrong = pick_wrong_experiment(pool, intended, expected_next, rng)
        segs.append({
            "video": wrong["video"],
            "trial_id": wrong["tid"],
            "protocol_id": wrong["proto"],
            "frame_indices": sample_halt_head(wrong["nframes"], halt_frames),
            "is_halt_window": True,
            "nframes": wrong["nframes"],
        })
        observed.append(wrong["proto"])
        negs.append(make_record(
            sample_id=f"neg_wrong_after_{i}_got{wrong['proto']}",
            intended=intended, segments=segs, observed=observed,
            label="halt", error_type="wrong_execution",
            reason=(
                f"Current protocol is wrong: expected P{i+1} (protocol {expected_next}) "
                f"but saw an unrelated experiment (protocol {wrong['proto']}, trial {wrong['tid']})."
            ),
            kind=f"neg_wrong_{i}_{wrong['proto']}",
        ))

    rng.shuffle(negs)
    negs = negs[:max_neg]
    expanded = []
    for rec in negs:
        expanded.extend(expand_halt_horizons(rec, max_h=min(5, halt_frames)))
    for _ in range(halt_oversample):
        for rec in expanded:
            samples.append(rec)
            counts["halt"] += 1
            counts[f"halt_{rec.get('error_type')}"] += 1
            counts[f"halt_h{rec.get('halt_horizon')}"] += 1

    rng.shuffle(samples)
    return samples, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
    ap.add_argument("--mistake-videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")
    ap.add_argument("--ann-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/FineBioQwenStream/data/proto_prefix_v1")
    ap.add_argument("--intended", default="1,2,3,4,5,6,7",
                    help="Ordered protocol ids for the session plan")
    ap.add_argument("--frames-per-proto", type=int, default=16,
                    help="Sparse frames from each *correct* protocol video")
    ap.add_argument("--halt-frames", type=int, default=5)
    ap.add_argument("--halt-oversample", type=int, default=3)
    ap.add_argument("--max-neg", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=20,
                    help="How many random trial combinations per split (data aug)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert 1 <= args.halt_frames <= 5
    intended = [int(x) for x in args.intended.split(",") if x.strip()]
    assert intended and all(p in PROTOCOL_NAMES for p in intended)

    out = Path(args.out_dir)
    videos_out = out / "videos"
    out.mkdir(parents=True, exist_ok=True)

    pool_all: dict[int, list[dict]] = defaultdict(list)
    for af in sorted(Path(args.ann_dir).glob("P*.txt")):
        tid, proto = parse_trial(af.name)
        if proto not in PROTOCOL_NAMES:
            continue
        src = resolve_video(tid, args.videos_dir, args.mistake_videos_dir)
        if src is None:
            continue
        fps, nframes = video_meta(src)
        if nframes <= 0:
            continue
        rel = link_video(src, videos_out, tid)
        pool_all[proto].append({
            "tid": tid, "proto": proto, "video": rel,
            "fps": fps, "nframes": nframes,
        })

    train_p, val_p, test_p = split_pool(pool_all, args.val_frac, args.test_frac, args.seed)

    meta_counts = {}
    for name, pool in [("train", train_p), ("val", val_p), ("test", test_p)]:
        all_samples = []
        total_counts = Counter()
        for r in range(args.repeats if name == "train" else max(2, args.repeats // 5)):
            samples, counts = build_split(
                pool, intended, args.frames_per_proto, args.halt_frames,
                args.halt_oversample if name == "train" else 1,
                args.max_neg, seed=args.seed + r + hash(name) % 1000,
            )
            # uniquify ids
            for s in samples:
                s["id"] = f"{name}_r{r}_{s['id']}"
            all_samples.extend(samples)
            total_counts.update(counts)
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for rec in all_samples:
                f.write(json.dumps(rec) + "\n")
        meta_counts[name] = {"n": len(all_samples), **dict(total_counts)}
        print(f"[{name}] samples={len(all_samples)} {dict(total_counts)}")

    meta = {
        "task": "finebio_protocol_prefix_stream_sft",
        "level": "protocol",  # NOT step
        "ssl": False,
        "intended_protocols": intended,
        "halt_frames_max": 5,
        "frames_per_proto": args.frames_per_proto,
        "early_detection": {
            "horizons": [1, 2, 3, 4, 5],
            "preference": "1 best, 2-4 ok, 5 must",
            "loss_weight": "w(k)=1.5/k + 1.0*[k==5]",
        },
        "loss": "Σ_k w(k) * (L_lm + λ_halt L_halt + λ_type L_type)_k",
        "error_types": ["missing_protocol", "wrong_execution"],
        "n_error_types": 2,
        "splits": meta_counts,
        "positive": "P1, P1+P2, ... (full protocol videos) → CONTINUE",
        "negative": {
            "missing_protocol": "skip intended protocol(s)",
            "wrong_execution": "insert wild wrong experiment video as current segment",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "protocol_reference.json").write_text(json.dumps(
        {str(p): PROTOCOL_NAMES[p] for p in PROTOCOL_IDS}, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
