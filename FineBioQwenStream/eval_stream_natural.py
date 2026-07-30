#!/usr/bin/env python3
"""Natural streaming rollout eval: video + protocol only (no stored frame_indices).

For each logical HALT event, walk k=1..5 where the stream so far is:
  [full protocol videos for correct prefix segs] + [first k raw frames of bad seg]
Good segments are FPS-sampled over the *entire* trial video (not the curated
4-frame indices from the jsonl). The bad segment's temporal content is only
what has arrived (frames 0..k-1); if that already fits max_frames we keep all,
else we downsample the concatenated stream uniformly.

CONTINUE prefixes: FPS-sample each full protocol video in the prefix.

This matches the deployment interface: inputs are video stream + intended plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from protocol_prompt import build_user_prompt, parse_decision


def _uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return [0]
    k = max(1, min(k, n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def sample_indices_at_fps(
    nframes: int,
    native_fps: float,
    sample_fps: float,
    max_per_seg: int | None = None,
) -> list[int]:
    native_fps = max(float(native_fps or 30.0), 1e-3)
    sample_fps = max(float(sample_fps), 1e-3)
    duration = nframes / native_fps
    k = max(1, int(round(duration * sample_fps)))
    if max_per_seg is not None:
        k = min(k, max(1, max_per_seg))
    k = min(k, max(nframes, 1))
    return _uniform_indices(nframes, k)


def load_video_range(
    path: str,
    *,
    end_exclusive: int | None = None,
    sample_fps: float,
    max_per_seg: int | None,
) -> np.ndarray:
    """Load frames from [0, end_exclusive) by FPS sampling (no curated indices)."""
    vr = VideoReader(path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    end = n if end_exclusive is None else max(1, min(int(end_exclusive), n))
    # Sample within the available prefix window only.
    native = float(vr.get_avg_fps() or 30.0)
    # Map FPS sampling onto the window length `end`.
    idxs_local = sample_indices_at_fps(end, native, sample_fps, max_per_seg)
    return vr.get_batch(idxs_local).asnumpy()


def configure_processor_pixels(processor, max_pixels: int, min_pixels: int):
    for proc in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if proc is None:
            continue
        if hasattr(proc, "max_pixels"):
            proc.max_pixels = max_pixels
        if hasattr(proc, "min_pixels"):
            proc.min_pixels = min_pixels
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            proc.size = {
                **proc.size,
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }


@torch.inference_mode()
def run_one(model, processor, frames, user_text, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user_text},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # No return_tensors/fps kwargs — let processor handle video tensors.
    raw = processor(text=[text], videos=[frames], padding=True)
    inputs = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            inputs[k] = torch.from_numpy(v).to(device)
        else:
            inputs[k] = torch.as_tensor(v).to(device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def base_id(sid: str) -> str | None:
    m = re.search(r"_h([1-5])$", sid)
    return sid[: m.start()] if m else None


def load_stream_frames(
    video_root: str,
    segments: list[dict],
    *,
    halt_k: int | None,
    sample_fps: float,
    max_frames: int,
    max_frames_per_seg: int | None,
) -> tuple[np.ndarray, dict]:
    """Build stream frames without using stored frame_indices.

    - Non-halt segments: FPS-sample the full video.
    - Halt window (last seg when halt_k set): only first halt_k raw frames exist;
      FPS-sample inside that window (usually keeps all k when k small).
    """
    parts = []
    info = {"seg_frames": [], "used_stored_indices": False}
    for si, seg in enumerate(segments):
        path = os.path.join(video_root, seg["video"])
        is_last = si == len(segments) - 1
        if halt_k is not None and is_last:
            # Stream content = only the first halt_k arrived frames of the bad video.
            # Keep all of them (they are the observation, not a curated subsample).
            vr = VideoReader(path, ctx=cpu(0), num_threads=2)
            n = len(vr)
            idxs = list(range(min(max(halt_k, 1), n)))
            frames = vr.get_batch(idxs).asnumpy()
        else:
            frames = load_video_range(
                path,
                end_exclusive=None,
                sample_fps=sample_fps,
                max_per_seg=max_frames_per_seg,
            )
        parts.append(frames)
        info["seg_frames"].append(int(len(frames)))

    frames = np.concatenate(parts, axis=0)
    info["n_before_cap"] = int(len(frames))
    if len(frames) > max_frames:
        keep = _uniform_indices(len(frames), max_frames)
        frames = frames[keep]
    info["n_after_cap"] = int(len(frames))
    return frames, info


def group_halt_events(rows: list[dict]) -> dict[str, dict]:
    """base_id -> canonical rec at h5 (full segment list + nframes on bad)."""
    events = {}
    for r in rows:
        if r.get("label") != "halt":
            continue
        bid = base_id(r["id"])
        if bid is None:
            continue
        h = r.get("halt_horizon")
        prev = events.get(bid)
        if prev is None or int(h or 0) > int(prev.get("halt_horizon") or 0):
            events[bid] = r
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="FPS used to sample full protocol videos (no curated indices)")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--max-frames-per-seg", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    ap.add_argument("--limit-events", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    scfg = {}
    if (ckpt / "config_task.json").exists():
        scfg = json.loads((ckpt / "config_task.json").read_text())
    base = scfg.get("model_path", args.model_path)

    processor = AutoProcessor.from_pretrained(
        str(ckpt) if (ckpt / "preprocessor_config.json").exists() else base,
        trust_remote_code=True,
    )
    configure_processor_pixels(processor, args.max_pixels, args.min_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()
    device = next(model.parameters()).device

    rows = []
    with open(args.data_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    events = group_halt_events(rows)
    continues = [r for r in rows if r.get("label") == "continue"]
    event_items = sorted(events.items())
    if args.limit_events:
        event_items = event_items[: args.limit_events]

    print(
        f"[natural-stream] events={len(event_items)} continue={len(continues)} "
        f"fps={args.sample_fps} max_frames={args.max_frames} "
        f"max_per_seg={args.max_frames_per_seg} ckpt={ckpt}",
        flush=True,
    )

    t0 = time.time()
    event_results = []
    exact = Counter()
    cdf = Counter()
    miss = 0
    latencies = []
    et_ok_at = []

    for ei, (bid, rec) in enumerate(event_items, 1):
        segs = rec["segments"]
        intended = rec["intended_protocols"]
        gt_et = rec.get("error_type")
        user_text = build_user_prompt(intended, n_segments_seen=len(segs))

        first = None
        first_et = None
        first_et_ok = False
        steps = []
        for k in range(1, 6):
            frames, info = load_stream_frames(
                args.video_root,
                segs,
                halt_k=k,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                max_frames_per_seg=args.max_frames_per_seg or None,
            )
            reply = run_one(model, processor, frames, user_text, device)
            pred, pred_et = parse_decision(reply)
            step = {
                "k": k,
                "pred": pred,
                "pred_et": pred_et,
                "reply": reply,
                "n_frames": int(len(frames)),
                "seg_frames": info["seg_frames"],
                "n_before_cap": info["n_before_cap"],
            }
            steps.append(step)
            if pred == "HALT" and first is None:
                first = k
                first_et = pred_et
                first_et_ok = (pred_et == gt_et) if pred_et and gt_et else False
                break

        if first is None:
            miss += 1
        else:
            exact[first] += 1
            latencies.append(first)
            et_ok_at.append(first_et_ok)
            for kk in range(first, 6):
                cdf[kk] += 1

        event_results.append({
            "id": bid,
            "gt_et": gt_et,
            "kind": rec.get("kind"),
            "n_segments": len(segs),
            "first_halt_k": first,
            "pred_et_at_halt": first_et,
            "et_ok_at_halt": first_et_ok,
            "steps": steps,
        })
        if ei % 5 == 0 or ei == len(event_items):
            det = len(latencies)
            print(
                f"[halt {ei}/{len(event_items)}] detected={det} miss={miss} "
                f"mean_lat={mean(latencies) if latencies else float('nan'):.2f} "
                f"{(time.time()-t0)/ei:.1f}s/event",
                flush=True,
            )

    # CONTINUE: natural FPS sample, no curated indices
    cont_rows = []
    fa = 0
    for i, rec in enumerate(continues, 1):
        frames, info = load_stream_frames(
            args.video_root,
            rec["segments"],
            halt_k=None,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            max_frames_per_seg=args.max_frames_per_seg or None,
        )
        user_text = build_user_prompt(
            rec["intended_protocols"], n_segments_seen=len(rec["segments"])
        )
        reply = run_one(model, processor, frames, user_text, device)
        pred, pred_et = parse_decision(reply)
        wrong = pred == "HALT"
        if wrong:
            fa += 1
        cont_rows.append({
            "id": rec["id"],
            "pred": pred,
            "pred_et": pred_et,
            "reply": reply,
            "n_frames": int(len(frames)),
            "seg_frames": info["seg_frames"],
            "false_halt": wrong,
        })
        if i % 10 == 0 or i == len(continues):
            print(
                f"[continue {i}/{len(continues)}] false_halt={fa} "
                f"{(time.time()-t0):.0f}s total",
                flush=True,
            )

    n = len(event_items)
    n_det = len(latencies)
    summary = {
        "mode": "natural_stream_rollout",
        "checkpoint": str(ckpt),
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames,
        "max_frames_per_seg": args.max_frames_per_seg,
        "n_halt_events": n,
        "detected_within_5": n_det,
        "detect_rate": n_det / max(n, 1),
        "miss_rate": miss / max(n, 1),
        "latency_mean": mean(latencies) if latencies else None,
        "latency_median": median(latencies) if latencies else None,
        "first_halt_hist": {str(k): exact[k] for k in range(1, 6)},
        "cdf_by_k": {str(k): cdf[k] / max(n, 1) for k in range(1, 6)},
        "et_acc_at_first_halt": (sum(et_ok_at) / len(et_ok_at)) if et_ok_at else None,
        "continue_n": len(continues),
        "continue_false_halt": fa,
        "continue_false_halt_rate": fa / max(len(continues), 1),
        "seconds": time.time() - t0,
        "note": (
            "No stored frame_indices. Good segs = FPS over full video; "
            "bad seg at step k = only first k arrived frames, then global max_frames cap."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": summary,
        "halt_events": event_results,
        "continue_rows": cont_rows,
    }, indent=2))

    print("\n===== NATURAL STREAM ROLLOUT =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"{'k':>3} {'first@k':>8} {'cdf≤k%':>8}")
    for k in range(1, 6):
        print(f"{k:3d} {exact[k]:8d} {100*cdf[k]/max(n,1):7.1f}%", flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
