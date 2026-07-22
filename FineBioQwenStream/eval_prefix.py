#!/usr/bin/env python3
"""Evaluate prefix SFT / zeroshot on val/test jsonl: decision + error_type accuracy.

Supports FPS resampling ablation via --sample-fps (ignore stored indices;
resample each full protocol segment at the given FPS, then cap).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from protocol_prompt import parse_decision


def _uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return [0]
    k = max(1, min(k, n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def load_frames(video_path: str, indices: list[int]):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(i, 0), n - 1) for i in indices]
    return vr.get_batch(idxs).asnumpy()


def sample_indices_at_fps(
    nframes: int,
    native_fps: float,
    sample_fps: float,
    max_per_seg: int | None = None,
) -> list[int]:
    """Uniformly sample the full clip at sample_fps; optionally cap then re-uniform."""
    native_fps = max(float(native_fps or 30.0), 1e-3)
    sample_fps = max(float(sample_fps), 1e-3)
    duration = nframes / native_fps
    k = max(1, int(round(duration * sample_fps)))
    if max_per_seg is not None:
        k = min(k, max(1, max_per_seg))
    k = min(k, max(nframes, 1))
    return _uniform_indices(nframes, k)


def load_sample_frames(
    video_root: str,
    rec: dict,
    max_frames: int,
    sample_fps: float | None = None,
    max_frames_per_seg: int | None = None,
) -> np.ndarray:
    """Load frames for one jsonl record.

    - sample_fps is None: use stored frame_indices (dataset default).
    - sample_fps set: resample each *full* protocol segment at that FPS.
      Halt windows (is_halt_window / ≤5 stored frames on last HALT seg) keep
      their stored early-detection indices so the label still matches.
    """
    if rec.get("segments"):
        parts = []
        segs = rec["segments"]
        for si, seg in enumerate(segs):
            path = os.path.join(video_root, seg["video"])
            stored = list(seg.get("frame_indices") or [])
            is_halt_win = bool(seg.get("is_halt_window")) or (
                rec.get("label") == "halt" and si == len(segs) - 1 and len(stored) <= 5
            )
            if sample_fps is None or is_halt_win:
                parts.append(load_frames(path, stored if stored else [0]))
            else:
                vr = VideoReader(path, ctx=cpu(0), num_threads=2)
                idxs = sample_indices_at_fps(
                    len(vr), float(vr.get_avg_fps() or 30.0), sample_fps, max_frames_per_seg,
                )
                parts.append(vr.get_batch(idxs).asnumpy())
        frames = np.concatenate(parts, axis=0)
    else:
        path = os.path.join(video_root, rec["video"])
        if sample_fps is None:
            frames = load_frames(path, rec["frame_indices"])
        else:
            vr = VideoReader(path, ctx=cpu(0), num_threads=2)
            idxs = sample_indices_at_fps(
                len(vr), float(vr.get_avg_fps() or 30.0), sample_fps, max_frames_per_seg,
            )
            frames = vr.get_batch(idxs).asnumpy()

    if len(frames) > max_frames:
        # Preserve temporal span: uniform downsample (not just tail crop)
        keep = _uniform_indices(len(frames), max_frames)
        frames = frames[keep]
    return frames


def configure_processor_pixels(processor, max_pixels: int, min_pixels: int):
    for proc in (getattr(processor, "image_processor", None),
                 getattr(processor, "video_processor", None)):
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def evaluate(
    model,
    processor,
    rows: list[dict],
    video_root: str,
    max_frames: int,
    sample_fps: float | None,
    max_frames_per_seg: int | None,
    device,
) -> dict:
    n = len(rows)
    dec_ok = et_ok = halt_n = 0
    conf = Counter()
    results = []
    t0 = time.time()
    n_frames_seen = []

    for i, rec in enumerate(rows):
        frames = load_sample_frames(
            video_root, rec, max_frames,
            sample_fps=sample_fps,
            max_frames_per_seg=max_frames_per_seg,
        )
        n_frames_seen.append(int(len(frames)))
        user_text = rec["messages"][0]["content"]
        reply = run_one(model, processor, frames, user_text, device)
        pred_d, pred_et = parse_decision(reply)
        gt_d = "CONTINUE" if rec["label"] == "continue" else "HALT"
        gt_et = rec.get("error_type")
        d_ok = pred_d == gt_d
        e_ok = True
        if gt_d == "HALT":
            halt_n += 1
            e_ok = (pred_et == gt_et) if pred_et and gt_et else False
            if e_ok:
                et_ok += 1
        if d_ok:
            dec_ok += 1
        conf[(gt_d, pred_d)] += 1
        results.append({
            "id": rec["id"], "gt": gt_d, "gt_et": gt_et,
            "pred": pred_d, "pred_et": pred_et, "reply": reply,
            "dec_ok": d_ok, "et_ok": e_ok, "n_frames": int(len(frames)),
        })
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(
                f"[{i+1}/{n}] fps={sample_fps} dec={dec_ok/(i+1):.3f} "
                f"et={et_ok/max(halt_n,1):.3f} "
                f"avg_frames={sum(n_frames_seen)/len(n_frames_seen):.1f} "
                f"{(time.time()-t0)/(i+1):.2f}s/sample",
                flush=True,
            )

    return {
        "summary": {
            "sample_fps": sample_fps,
            "max_frames": max_frames,
            "max_frames_per_seg": max_frames_per_seg,
            "n": n,
            "decision_accuracy": dec_ok / max(n, 1),
            "error_type_accuracy_on_halt": et_ok / max(halt_n, 1),
            "n_halt_gt": halt_n,
            "avg_frames": sum(n_frames_seen) / max(len(n_frames_seen), 1),
            "confusion_gt_pred": {f"{a}->{b}": c for (a, b), c in conf.items()},
            "seconds": time.time() - t0,
        },
        "rows": results,
    }


def load_model(args):
    if args.zero_shot:
        base = args.model_path
        ckpt = None
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="sdpa",
        )
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint required unless --zero-shot")
        ckpt = Path(args.checkpoint)
        scfg = json.loads((ckpt / "config_task.json").read_text()) if (ckpt / "config_task.json").exists() else {}
        base = scfg.get("model_path", args.model_path)
        if "max_frames" in scfg and args.max_frames == 8:
            args.max_frames = int(scfg["max_frames"])
        if "max_pixels" in scfg:
            args.max_pixels = int(scfg["max_pixels"])
        if "min_pixels" in scfg:
            args.min_pixels = int(scfg["min_pixels"])
        processor = AutoProcessor.from_pretrained(
            str(ckpt) if (ckpt / "preprocessor_config.json").exists() else base,
            trust_remote_code=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="sdpa",
        )
        if (ckpt / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ckpt))

    configure_processor_pixels(processor, args.max_pixels, args.min_pixels)
    model.eval()
    return model, processor, base, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA/SFT dir; omit with --zero-shot")
    ap.add_argument("--zero-shot", action="store_true",
                    help="Evaluate base model without LoRA adapters")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--max-frames-per-seg", type=int, default=0,
                    help="Cap per protocol segment before concat (0=unlimited)")
    ap.add_argument("--sample-fps", type=float, default=None,
                    help="If set, resample full protocol segments at this FPS")
    ap.add_argument("--fps-grid", type=str, default=None,
                    help="Comma list for ablation, e.g. 0.25,0.5,1,2,4 "
                         "(runs all; also includes stored-index baseline as fps=null)")
    ap.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, processor, base, ckpt = load_model(args)
    device = next(model.parameters()).device

    rows = []
    with open(args.data_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    max_per_seg = args.max_frames_per_seg if args.max_frames_per_seg > 0 else None

    if args.fps_grid:
        grid = []
        for tok in args.fps_grid.split(","):
            tok = tok.strip().lower()
            if tok in {"", "none", "null", "stored"}:
                grid.append(None)
            else:
                grid.append(float(tok))
        # always put stored-index baseline first if not present
        if None not in grid:
            grid = [None] + grid

        out_dir = Path(args.out) if args.out else Path(
            "/scratch/ll5914/Labos/FineBioQwenStream/outputs/fps_ablation_zeroshot"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        table = []
        print(
            f"[ablation] n={len(rows)} fps_grid={grid} max_frames={args.max_frames} "
            f"max_frames_per_seg={max_per_seg}",
            flush=True,
        )
        for fps in grid:
            tag = "stored" if fps is None else f"fps{fps:g}"
            print(f"\n===== zeroshot {tag} =====", flush=True)
            pack = evaluate(
                model, processor, rows, args.video_root,
                args.max_frames, fps, max_per_seg, device,
            )
            pack["summary"]["mode"] = "zeroshot" if args.zero_shot else "finetuned"
            pack["summary"]["base_model"] = base
            pack["summary"]["max_pixels"] = args.max_pixels
            (out_dir / f"eval_{tag}.json").write_text(json.dumps(pack, indent=2))
            s = pack["summary"]
            table.append({
                "sample_fps": s["sample_fps"],
                "decision_accuracy": s["decision_accuracy"],
                "error_type_accuracy_on_halt": s["error_type_accuracy_on_halt"],
                "avg_frames": s["avg_frames"],
                "seconds": s["seconds"],
                "confusion_gt_pred": s["confusion_gt_pred"],
            })
            print(json.dumps(s, indent=2), flush=True)

        summary_path = out_dir / "ablation_summary.json"
        summary_path.write_text(json.dumps({
            "base_model": base,
            "n": len(rows),
            "max_frames": args.max_frames,
            "max_frames_per_seg": max_per_seg,
            "max_pixels": args.max_pixels,
            "grid": table,
        }, indent=2))
        print("\n===== FPS ablation summary =====", flush=True)
        print(
            f"{'fps':>10} {'dec_acc':>10} {'et_acc':>10} {'avg_fr':>8} {'sec':>8}",
            flush=True,
        )
        for row in table:
            fps = "stored" if row["sample_fps"] is None else f"{row['sample_fps']:g}"
            print(
                f"{fps:>10} {row['decision_accuracy']:10.3f} "
                f"{row['error_type_accuracy_on_halt']:10.3f} "
                f"{row['avg_frames']:8.1f} {row['seconds']:8.1f}",
                flush=True,
            )
        print(f"[done] {summary_path}", flush=True)
        return

    # single-run mode
    print(
        f"[eval] n={len(rows)} mode={'zeroshot' if args.zero_shot else ckpt} "
        f"sample_fps={args.sample_fps} max_frames={args.max_frames}",
        flush=True,
    )
    pack = evaluate(
        model, processor, rows, args.video_root,
        args.max_frames, args.sample_fps, max_per_seg, device,
    )
    pack["summary"]["mode"] = "zeroshot" if args.zero_shot else "finetuned"
    pack["summary"]["checkpoint"] = None if args.zero_shot else str(ckpt)
    pack["summary"]["base_model"] = base
    pack["summary"]["max_pixels"] = args.max_pixels

    if args.out:
        out = Path(args.out)
    elif args.zero_shot:
        out = Path("/scratch/ll5914/Labos/FineBioQwenStream/outputs") / "eval_val_zeroshot.json"
    else:
        out = ckpt / "eval_prefix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, indent=2))
    print(json.dumps(pack["summary"], indent=2), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
