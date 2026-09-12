#!/usr/bin/env python3
"""Evaluate the oracle-triggered 32B expert with dataset history/current step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval_egoproactive_bio_expert_oracle import score_semantics, summarize
from models import QwenVL
from pipeline import load_indices, reproject_keep
from prompts import expert_prompt, parse_expert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--tau-reproj", type=float, default=0.12)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in args.eval_jsonl.read_text().splitlines()]
    if len(cases) != 14:
        raise ValueError("Expected 14 contextual oracle-expert cases")
    model = QwenVL(args.expert, max_new_tokens=96)
    rows = []
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for case in cases:
            original = load_indices(case["video"], case["frame_indices"])
            kept, keep_local = reproject_keep(original, tau=args.tau_reproj)
            variants = [
                ("official_raw_reprojection", case["history_official_raw"], kept),
                ("official_raw_full_16", case["history_official_raw"], original),
                ("official_clean_reprojection", case["history_official_clean"], kept),
                ("official_clean_full_16", case["history_official_clean"], original),
                ("full_raw_reprojection", case["history_full_raw"], kept),
                ("full_clean_reprojection", case["history_full_clean"], kept),
            ]
            prompt = expert_prompt(
                case["protocol"], case["asr_text"],
                judger_reason=case["oracle_reason"], current_step=case["current_step"],
            )
            for variant, history, frames in variants:
                raw, latency = model.generate(frames, prompt, history=history)
                pred_type, message = parse_expert(raw)
                semantic = score_semantics(message, case["semantic_rubric"])
                row = {
                    "id": case["id"], "variant": variant,
                    "oracle_reason_sent": case["oracle_reason"],
                    "current_step_sent": case["current_step"], "history_sent": history,
                    "gt_type": case["gt_type"], "pred_type": pred_type,
                    "type_correct": pred_type.lower() == case["gt_type"],
                    "message": message, "raw": raw, "prompt": prompt,
                    "reference_guidance": case["reference_guidance"],
                    "semantic_rubric": case["semantic_rubric"], "semantic": semantic,
                    "n_frames_original": len(original), "n_frames_sent": len(frames),
                    "keep_local": keep_local, "latency_s": latency,
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(case["id"], variant, pred_type, message, flush=True)
    summary = summarize(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "run.json").write_text(json.dumps({
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expert": args.expert, "tau_reproj": args.tau_reproj,
        "eval_jsonl": str(args.eval_jsonl), "n_model_calls": len(rows),
        "oracle_disclosure": "oracle reason plus exact dataset task/current_step",
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
