#!/usr/bin/env python3
"""Small-model probes that separate prompt following from visual gating."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from models import QwenVL
from pipeline import load_indices, reproject_keep
from prompts import judger_prompt, parse_judger


def explicit_probe_text(clip: dict) -> str:
    return {
        "safety": "I see a current safety issue. Please check it now.",
        "action_error": "I made an action error that conflicts with the protocol.",
        "next_step": "I have completed the current step. What should I do next?",
        "none": "",
    }[clip["gt_reason"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judger", required=True)
    parser.add_argument("--tau-reproj", type=float, default=0.12)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = [json.loads(line) for line in args.eval_jsonl.read_text().splitlines() if line.strip()]
    model = QwenVL(args.judger, max_new_tokens=24)
    rows = []
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for clip in clips:
            original = load_indices(clip["video"], clip["frame_indices"])
            kept, keep = reproject_keep(original, tau=args.tau_reproj)
            variants = [
                ("native_original_prompt", kept, "", False),
                ("native_bio_prompt", kept, "", True),
                ("native_bio_no_reprojection", original, "", True),
                ("native_bio_black", np.zeros_like(kept), "", True),
                ("native_bio_explicit_text_probe", kept, explicit_probe_text(clip), True),
                (
                    "native_original_question_probe",
                    kept,
                    "What should I do next according to the protocol?",
                    False,
                ),
            ]
            for variant, frames, speech, proactive in variants:
                prompt = judger_prompt(
                    clip["protocol"], speech, proactive_next_step=proactive
                )
                raw, latency = model.generate(frames, prompt)
                fired, reason = parse_judger(raw)
                row = {
                    "id": clip["id"], "variant": variant,
                    "gt_fire": clip["gt_fire"], "gt_reason": clip["gt_reason"],
                    "fired": fired, "reason": reason, "raw": raw,
                    "asr_text": speech, "prompt": prompt,
                    "n_frames_before_reprojection": len(original),
                    "n_frames_after_reprojection": len(kept), "keep_local": keep,
                    "latency_judger_s": latency,
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(clip["id"], variant, repr(raw), flush=True)
    summary = {}
    for variant in dict.fromkeys(row["variant"] for row in rows):
        part = [row for row in rows if row["variant"] == variant]
        tp = sum(row["gt_fire"] and row["fired"] for row in part)
        fp = sum(not row["gt_fire"] and row["fired"] for row in part)
        fn = sum(row["gt_fire"] and not row["fired"] for row in part)
        tn = sum(not row["gt_fire"] and not row["fired"] for row in part)
        summary[variant] = {
            "n": len(part), "fired": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "raw_counts": dict(Counter(row["raw"] for row in part)),
            "reason_counts": dict(Counter(row["reason"] or "none" for row in part)),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
