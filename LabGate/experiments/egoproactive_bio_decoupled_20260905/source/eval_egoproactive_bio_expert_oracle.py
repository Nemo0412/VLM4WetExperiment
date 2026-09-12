#!/usr/bin/env python3
"""Evaluate only the 32B expert under an oracle-positive gate."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from models import QwenVL
from pipeline import load_indices, reproject_keep
from prompts import expert_prompt, parse_expert


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_semantics(message: str, rubric: list[list[str]]) -> dict:
    text = normalize(message)
    covered = [any(normalize(term) in text for term in group) for group in rubric]
    return {
        "covered_groups": covered,
        "groups_covered": sum(covered),
        "groups_total": len(covered),
        "concept_recall": sum(covered) / max(len(covered), 1),
        "all_required_concepts": all(covered),
    }


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for variant in dict.fromkeys(row["variant"] for row in rows):
        part = [row for row in rows if row["variant"] == variant]
        by_type = {}
        for gt_type in sorted({row["gt_type"] for row in part}):
            typed = [row for row in part if row["gt_type"] == gt_type]
            by_type[gt_type] = {
                "n": len(typed),
                "type_accuracy": sum(row["type_correct"] for row in typed) / len(typed),
                "semantic_complete_rate": sum(row["semantic"]["all_required_concepts"] for row in typed) / len(typed),
            }
        summaries[variant] = {
            "n": len(part),
            "type_accuracy": sum(row["type_correct"] for row in part) / len(part),
            "semantic_complete_rate": sum(row["semantic"]["all_required_concepts"] for row in part) / len(part),
            "mean_concept_recall": sum(row["semantic"]["concept_recall"] for row in part) / len(part),
            "nonempty_message_rate": sum(bool(row["message"].strip()) for row in part) / len(part),
            "pred_type_counts": dict(Counter(row["pred_type"] for row in part)),
            "by_gt_type": by_type,
            "mean_frames_sent": sum(row["n_frames_sent"] for row in part) / len(part),
            "mean_latency_s": sum(row["latency_s"] for row in part) / len(part),
        }
    return summaries


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
    if len(cases) != 14 or any(not case["oracle_reason"] for case in cases):
        raise ValueError("Expected 14 oracle-positive expert cases")
    model = QwenVL(args.expert, max_new_tokens=96)
    rows = []
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for case in cases:
            original = load_indices(case["video"], case["frame_indices"])
            kept, keep_local = reproject_keep(original, tau=args.tau_reproj)
            variants = [
                ("native_reprojection_oracle_reason", kept, case["oracle_reason"]),
                ("full_16_frames_oracle_reason", original, case["oracle_reason"]),
                ("native_reprojection_trigger_only", kept, "unknown"),
                ("full_16_frames_trigger_only", original, "unknown"),
            ]
            for variant, frames, reason in variants:
                prompt = expert_prompt(case["protocol"], case["asr_text"], judger_reason=reason)
                raw, latency = model.generate(frames, prompt)
                pred_type, message = parse_expert(raw)
                semantic = score_semantics(message, case["semantic_rubric"])
                row = {
                    "id": case["id"], "variant": variant,
                    "oracle_reason_sent": reason, "gt_type": case["gt_type"],
                    "pred_type": pred_type, "type_correct": pred_type.lower() == case["gt_type"],
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
        "oracle_disclosure": "oracle_reason is scoring metadata deliberately supplied as the assumed correct small-model output",
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

