#!/usr/bin/env python3
"""Evaluate only the 3B gate on the decoupled Bio gate set."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from models import QwenVL
from pipeline import load_indices, reproject_keep
from prompts import judger_prompt, parse_judger


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for variant in dict.fromkeys(row["variant"] for row in rows):
        part = [row for row in rows if row["variant"] == variant]
        tp = sum(row["gt_fire"] and row["fired"] for row in part)
        fp = sum(not row["gt_fire"] and row["fired"] for row in part)
        fn = sum(row["gt_fire"] and not row["fired"] for row in part)
        tn = sum(not row["gt_fire"] and not row["fired"] for row in part)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        positive = [row for row in part if row["gt_fire"]]
        summaries[variant] = {
            "n": len(part),
            "accuracy": (tp + tn) / len(part),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fire_rate": (tp + fp) / len(part),
            "reason_accuracy_on_positive": sum(
                row["reason"] == row["gt_reason"] for row in positive
            ) / len(positive),
            "raw_counts": dict(Counter(row["raw"] for row in part)),
            "mean_frames_sent": sum(row["n_frames_sent"] for row in part) / len(part),
            "mean_latency_s": sum(row["latency_s"] for row in part) / len(part),
        }
    return summaries


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
    cases = [json.loads(line) for line in args.eval_jsonl.read_text().splitlines()]
    if len(cases) != 25:
        raise ValueError("Expected 25 gate cases")
    model = QwenVL(args.judger, max_new_tokens=24)
    rows = []
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for case in cases:
            original = load_indices(case["video"], case["frame_indices"])
            kept, keep_local = reproject_keep(original, tau=args.tau_reproj)
            for variant, frames in [("native_reprojection", kept), ("full_16_frames", original)]:
                prompt = judger_prompt(case["protocol"], case["asr_text"], proactive_next_step=True)
                raw, latency = model.generate(frames, prompt)
                fired, reason = parse_judger(raw)
                row = {
                    "id": case["id"], "variant": variant,
                    "gt_fire": case["gt_fire"], "gt_reason": case["gt_reason"],
                    "fired": fired, "reason": reason, "raw": raw,
                    "prompt": prompt, "n_frames_original": len(original),
                    "n_frames_sent": len(frames), "keep_local": keep_local,
                    "latency_s": latency,
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(case["id"], variant, repr(raw), flush=True)
    summary = summarize(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "run.json").write_text(json.dumps({
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judger": args.judger, "tau_reproj": args.tau_reproj,
        "eval_jsonl": str(args.eval_jsonl), "n_model_calls": len(rows),
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

