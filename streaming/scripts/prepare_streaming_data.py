#!/usr/bin/env python3
"""Build FineBioStreaming v2 chunk dataset.

v2 changes vs v1:
  - Prompt includes explicit protocol progress history (prior steps / chunks).
  - HALT is labeled at the first observable error chunk (not only the last prefix).
  - Real mistakes with annotations: halt = first divergence vs a same-protocol reference.
  - Major mistake videos without anns: time-aligned to a correct sibling + known error window.
  - Mild oversampling of HALT prefixes to reduce CONTINUE collapse.
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

# (error_type, reason, optional halt hint)
REAL_MISTAKES = {
    "P06_03_02": ("missing_step", "Missing step: sterile water wash is skipped.", "add_sterile_water"),
    "P11_06_01": ("redundant_step", "Redundant step: an extra wash buffer is performed.", "add_wash_buffer"),
    "P17_02_02": ("redundant_step", "Redundant step: an extra PBS wash is performed.", "add_pbs"),
    "P03_03_02": ("within_step_error", "Within-step error: no pipetting during suction.", "aspirate_supernatant"),
    "P03_04_02": ("within_step_error", "Within-step error: no pipetting during suction.", "aspirate_supernatant"),
    "P07_04_01": ("within_step_error", "Within-step error: the vortex step is skipped.", "vortex"),
    "P17_07_01": ("wrong_order", "Wrong order: dispense and detach spin column are swapped.", None),
    "P18_02_01": ("within_step_error", "Within-step error / missing end: spindown is skipped.", "spindown"),
    "P18_05_01": ("missing_step", "Missing step: last step missing due to missing frames.", None),
    "P24_07_01": ("missing_step", "Missing step: dispense after second-to-last spindown is skipped.", "dispense_solution"),
    "P28_06_02": ("wrong_order", "Wrong order: dispense and detach spin column are swapped.", None),
}

# Correct sibling used to time-align major mistakes that lack annotations.
MAJOR_SIBLING = {
    "P06_03_02": "P06_03_01",
    "P17_02_02": "P17_02_01",
    "P11_06_01": "P10_06_01",
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
    s = re.sub(r"\bpbs\b", "PBS", s, flags=re.I)
    s = re.sub(r"\bdna\b", "DNA", s, flags=re.I)
    s = re.sub(r"\bpcr\b", "PCR", s, flags=re.I)
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
        Path(mistake_dir) / f"{tid}.mp4",
        Path(videos_dir) / f"{tid}.mp4",
        Path(videos_dir) / "finebio_videos_w640" / f"{tid}.mp4",
    ):
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


def build_fixed_chunks(duration: float, fps: float, nframes: int, window_sec: float, k: int) -> list[dict]:
    chunks = []
    t = 0.0
    while t < duration - 1e-6:
        t1 = min(t + window_sec, duration)
        idxs = sample_indices_in_span(t, t1, fps, nframes, k)
        chunks.append({
            "chunk_id": len(chunks),
            "t0": t,
            "t1": t1,
            "steps": [f"window_{len(chunks)}"],
            "expected_step": f"window_{len(chunks)}",
            "frame_indices": idxs,
        })
        t += window_sec
    return chunks


def corrupt_drop(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str, str]:
    if len(spans) < 3:
        return spans, -1, "", "", ""
    drop_i = random.randint(1, len(spans) - 2)
    dropped = spans[drop_i][2]
    new_spans = spans[:drop_i] + spans[drop_i + 1 :]
    # First chunk where observer should notice the missing step
    halt_step_idx = min(drop_i, len(new_spans) - 1)
    expected = dropped
    observed = new_spans[halt_step_idx][2]
    reason = (
        f"Missing step: expected '{humanize(expected)}' next, "
        f"but observed '{humanize(observed)}' instead."
    )
    return new_spans, halt_step_idx, "missing_step", reason, expected


def corrupt_insert(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str, str]:
    if len(spans) < 3:
        return spans, -1, "", "", ""
    ins_i = random.randint(1, len(spans) - 2)
    dup = spans[ins_i]
    new_spans = spans[: ins_i + 1] + [dup] + spans[ins_i + 1 :]
    halt_step_idx = ins_i + 1
    expected = spans[ins_i + 1][2] if ins_i + 1 < len(spans) else spans[-1][2]
    observed = dup[2]
    reason = (
        f"Redundant step: expected '{humanize(expected)}' next, "
        f"but '{humanize(observed)}' is repeated."
    )
    return new_spans, halt_step_idx, "redundant_step", reason, expected


def corrupt_shuffle(spans: list[tuple[float, float, str]]) -> tuple[list, int, str, str, str]:
    if len(spans) < 3:
        return spans, -1, "", "", ""
    new_spans = spans.copy()
    i, j = sorted(random.sample(range(1, len(spans) - 1), 2))
    new_spans[i], new_spans[j] = new_spans[j], new_spans[i]
    halt_step_idx = i
    expected = spans[i][2]
    observed = new_spans[i][2]
    reason = (
        f"Wrong order: expected '{humanize(expected)}' but observed "
        f"'{humanize(observed)}'."
    )
    return new_spans, halt_step_idx, "wrong_order", reason, expected


def format_history(prefix_steps: list[str]) -> str:
    if not prefix_steps:
        return "(none yet)"
    return " -> ".join(humanize(s) for s in prefix_steps)


def build_conv(
    proto: int,
    label: str,
    reason: str,
    history_steps: list[str],
    current_steps: list[str],
    expected_next: str = "",
) -> list[dict]:
    hist = format_history(history_steps)
    cur = ", ".join(humanize(s) for s in current_steps) if current_steps else "(unclear)"
    expect_line = ""
    if expected_next and label != "continue":
        expect_line = f" According to the protocol, the next expected step was '{humanize(expected_next)}'."
    elif expected_next:
        expect_line = f" Next expected step: '{humanize(expected_next)}'."

    q = (
        "<image>\nYou are monitoring a wet-lab experiment in real time.\n"
        f"Intended protocol {proto}: {PROTOCOL_NAMES[proto]}.\n"
        f"Progress so far: {hist}.\n"
        f"Current video chunk shows: {cur}.{expect_line}\n"
        "Decide CONTINUE or HALT. If HALT, state error_type and a brief reason."
    )
    if label == "continue":
        a = "CONTINUE. Observed steps still match the protocol."
    else:
        a = f"HALT. error_type={label}. {reason}"
    return [{"from": "human", "value": q}, {"from": "gpt", "value": a}]


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
    expected_next_at_halt: str = "",
) -> list[dict]:
    samples = []
    max_k = halt_chunk_id if halt_chunk_id >= 0 else len(chunks) - 1
    for k in range(max_k + 1):
        is_halt = halt_chunk_id >= 0 and k == halt_chunk_id
        label = error_type if is_halt else "continue"
        prefix = chunks[: k + 1]
        last = prefix[-1]
        history_steps = []
        for c in prefix[:-1]:
            history_steps.extend(c["steps"])
        expected_next = expected_next_at_halt if is_halt else (last.get("expected_step") or "")
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
            "expected_next": expected_next,
            "halt_label": HALT_LABELS.get(label, 0),
            "halt_name": label,
            "reason": reason if is_halt else "",
            "conversations": build_conv(
                proto, label, reason if is_halt else "",
                history_steps, last["steps"], expected_next,
            ),
            "prefix_steps": history_steps + last["steps"],
            "history_steps": history_steps,
            "current_steps": last["steps"],
        })
    return samples


def pick_reference_spans(proto: int, by_proto: dict[int, list], exclude: str) -> list | None:
    cands = [sp for tid, sp in by_proto.get(proto, []) if tid != exclude and tid not in REAL_MISTAKES]
    if not cands:
        cands = [sp for tid, sp in by_proto.get(proto, []) if tid != exclude]
    if not cands:
        return None
    # Prefer median-length sequence
    cands = sorted(cands, key=len)
    return cands[len(cands) // 2]


def first_divergence(obs: list[str], ref: list[str]) -> int | None:
    n = min(len(obs), len(ref))
    for i in range(n):
        if obs[i] != ref[i]:
            return i
    if len(obs) != len(ref):
        return n
    return None


def halt_index_for_real(
    tid: str,
    spans: list[tuple[float, float, str]],
    ref_spans: list[tuple[float, float, str]] | None,
) -> tuple[int, str]:
    """Return (halt_step_idx, expected_next)."""
    et, reason, hint = REAL_MISTAKES[tid]
    obs = [t for _, _, t in spans]
    ref = [t for _, _, t in ref_spans] if ref_spans else []

    div = first_divergence(obs, ref) if ref else None
    if div is not None and div < len(spans):
        expected = ref[div] if div < len(ref) else (hint or obs[min(div, len(obs) - 1)])
        return div, expected

    # within-step / missing with identical task lists: use hint step occurrence
    if hint:
        for i, t in enumerate(obs):
            if t == hint:
                # For missing sterile water etc. on major vids handled elsewhere;
                # for within-step, halt at the hinted step.
                if et == "missing_step" and tid in ("P18_05_01",):
                    return max(len(obs) - 1, 0), hint
                return i, hint
        # hint not in obs => missing; halt where it should have appeared in ref
        if ref and hint in ref:
            return min(ref.index(hint), len(obs) - 1), hint

    # fallback: last step
    return max(len(spans) - 1, 0), hint or (obs[-1] if obs else "")


def major_mistake_halt_chunk(tid: str, chunks: list[dict], sibling_spans: list) -> tuple[int, str]:
    """Map known major error onto fixed/sibling-aligned chunks."""
    et, reason, hint = REAL_MISTAKES[tid]
    expected = hint or ""
    if sibling_spans and hint:
        # time of first hint occurrence on sibling, scaled into mistake duration
        sib_t = next((s for s, e, t in sibling_spans if t == hint), None)
        if sib_t is not None:
            # For redundant: second occurrence
            if et == "redundant_step":
                hits = [s for s, e, t in sibling_spans if t == hint]
                if len(hits) >= 2:
                    sib_t = hits[1]
                elif hits:
                    sib_t = hits[0]
            dur_sib = sibling_spans[-1][1]
            dur_mis = chunks[-1]["t1"]
            t_mis = sib_t * (dur_mis / max(dur_sib, 1e-6))
            for i, ch in enumerate(chunks):
                if ch["t1"] >= t_mis:
                    return i, expected
    # fallback windows used in eval
    defaults = {
        "P06_03_02": 200.0,
        "P17_02_02": 55.0,
        "P11_06_01": 100.0,
    }
    t0 = defaults.get(tid, chunks[len(chunks) // 2]["t0"])
    for i, ch in enumerate(chunks):
        if ch["t1"] >= t0:
            return i, expected
    return max(len(chunks) // 2, 0), expected


def balance_halt(samples: list[dict], halt_repeat: int) -> list[dict]:
    if halt_repeat <= 1:
        return samples
    out = list(samples)
    for s in samples:
        if int(s["halt_label"]) != 0:
            out.extend([s] * (halt_repeat - 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640")
    ap.add_argument("--ann-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations")
    ap.add_argument("--mistake-videos-dir", default="/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/FineBioStreaming/data/streaming_v2")
    ap.add_argument("--frames-per-chunk", type=int, default=8)
    ap.add_argument("--steps-per-chunk", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--synth-per-trial", type=int, default=3)
    ap.add_argument("--halt-repeat", type=int, default=3, help="Oversample HALT prefixes")
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

    all_spans: dict[str, list] = {}
    by_proto: dict[int, list] = defaultdict(list)
    for af in ann_files:
        tid, proto = parse_trial_id(af.name)
        if proto not in PROTOCOL_NAMES:
            continue
        spans = step_spans(read_segments(af))
        if len(spans) < 2:
            continue
        all_spans[tid] = spans
        by_proto[proto].append((tid, spans))

    step_counter: Counter = Counter()
    for spans in all_spans.values():
        for _, _, t in spans:
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

    # ---- annotated trials (intact + real mistakes with anns + synth) ----
    for tid, spans in all_spans.items():
        _, proto = parse_trial_id(tid)
        video = resolve_video(tid, args.videos_dir, args.mistake_videos_dir)
        if video is None:
            skipped.append(tid)
            continue
        fps, nframes = video_meta(video)
        if nframes <= 0 or len(spans) < 3:
            skipped.append(tid)
            continue

        rel = link_video(video, videos_out, tid)
        pid = PROTO_TO_CLASS[proto]
        is_real = tid in REAL_MISTAKES

        intact_chunks = build_chunks_from_spans(
            spans, fps, nframes, args.frames_per_chunk, args.steps_per_chunk)

        if is_real:
            ref = pick_reference_spans(proto, by_proto, tid)
            halt_step, expected_next = halt_index_for_real(tid, spans, ref)
            halt_chunk = min(halt_step // args.steps_per_chunk, len(intact_chunks) - 1)
            et, reason, _ = REAL_MISTAKES[tid]
            for s in emit_prefix_samples(
                tid, rel, proto, pid, intact_chunks, halt_chunk, et, reason,
                integrity=0, corruption="real_mistake",
                expected_next_at_halt=expected_next,
            ):
                s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
                counts[et if s["halt_name"] != "continue" else "continue"] += 1
                counts["real"] += 1
                samples.append(s)
            continue

        # Intact CONTINUE prefixes
        for s in emit_prefix_samples(
            tid, rel, proto, pid, intact_chunks, -1, "continue", "",
            integrity=1, corruption=None,
        ):
            s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
            counts["continue"] += 1
            samples.append(s)

        # Synthetic corruptions on intact trials
        for k in range(args.synth_per_trial):
            mode, fn = corruptors[k % len(corruptors)]
            new_spans, halt_step_idx, et, reason, expected_next = fn(spans)
            if halt_step_idx < 0 or not et:
                continue
            chunks = build_chunks_from_spans(
                new_spans, fps, nframes, args.frames_per_chunk, args.steps_per_chunk)
            halt_chunk = min(halt_step_idx // args.steps_per_chunk, len(chunks) - 1)
            for s in emit_prefix_samples(
                tid, rel, proto, pid, chunks, halt_chunk, et, reason,
                integrity=0, corruption=mode,
                expected_next_at_halt=expected_next,
            ):
                s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
                counts[et if s["halt_name"] != "continue" else "continue"] += 1
                counts[f"synth_{mode}"] += 1
                samples.append(s)

    # ---- major mistake videos without annotations ----
    for tid, sibling in MAJOR_SIBLING.items():
        if tid in all_spans:
            continue  # already handled via anns
        video = resolve_video(tid, args.videos_dir, args.mistake_videos_dir)
        if video is None:
            skipped.append(tid)
            continue
        fps, nframes = video_meta(video)
        if nframes <= 0:
            skipped.append(tid)
            continue
        duration = nframes / max(fps, 1e-6)
        _, proto = parse_trial_id(tid)
        if proto not in PROTOCOL_NAMES:
            # P11_06 is protocol 6
            proto = int(re.match(r"P\d+_(\d+)_", tid).group(1))
        rel = link_video(video, videos_out, tid)
        pid = PROTO_TO_CLASS[proto]
        chunks = build_fixed_chunks(
            duration, fps, nframes, args.window_sec, args.frames_per_chunk)

        # Attach sibling step names into window chunks by time scaling for richer history
        sib_spans = all_spans.get(sibling, [])
        if sib_spans:
            dur_sib = sib_spans[-1][1]
            scale = duration / max(dur_sib, 1e-6)
            for ch in chunks:
                mid = 0.5 * (ch["t0"] + ch["t1"])
                # map mid back to sibling time
                sib_t = mid / max(scale, 1e-6)
                near = min(sib_spans, key=lambda x: abs(0.5 * (x[0] + x[1]) - sib_t))
                ch["steps"] = [near[2]]
                ch["expected_step"] = near[2]

        halt_chunk, expected_next = major_mistake_halt_chunk(tid, chunks, sib_spans)
        et, reason, _ = REAL_MISTAKES[tid]
        for s in emit_prefix_samples(
            tid, rel, proto, pid, chunks, halt_chunk, et, reason,
            integrity=0, corruption="real_major",
            expected_next_at_halt=expected_next or "",
        ):
            s["expected_step_id"] = step_vocab.get(s["expected_step"], -100)
            counts[et if s["halt_name"] != "continue" else "continue"] += 1
            counts["real_major"] += 1
            samples.append(s)

    samples = balance_halt(samples, args.halt_repeat)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val, train = samples[:n_val], samples[n_val:]
    meta = {
        "task": "finebio_streaming_halt_v2",
        "version": 2,
        "frames_per_chunk": args.frames_per_chunk,
        "steps_per_chunk": args.steps_per_chunk,
        "window_sec": args.window_sec,
        "synth_per_trial": args.synth_per_trial,
        "halt_repeat": args.halt_repeat,
        "n_step_vocab": len(step_vocab),
        "halt_labels": HALT_LABELS,
        "features": [
            "history_in_prompt",
            "halt_at_first_error_chunk",
            "real_mistake_divergence",
            "major_mistake_time_align",
            "halt_oversample",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps(val))

    n_halt = sum(1 for s in samples if s["halt_label"] != 0)
    n_cont = sum(1 for s in samples if s["halt_label"] == 0)
    print(f"[done] annotated_trials={len(all_spans)} skipped={len(skipped)}")
    print(f"  samples={len(samples)} train={len(train)} val={len(val)}")
    print(f"  continue={n_cont} halt={n_halt} ratio_halt={n_halt / max(len(samples), 1):.3f}")
    print(f"  counts={dict(counts)}")
    print(f"  step_vocab={len(step_vocab)} -> {out}")


if __name__ == "__main__":
    main()
