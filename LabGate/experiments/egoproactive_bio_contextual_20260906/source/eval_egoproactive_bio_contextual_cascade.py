#!/usr/bin/env python3
"""Run the real 3B-to-32B cascade with dataset history and current-step context."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import torch

from eval_egoproactive_bio_expert_oracle import score_semantics
from models import QwenVL
from pipeline import load_indices, reproject_keep
from prompts import expert_prompt, judger_prompt, parse_expert, parse_judger


VARIANTS = (
    ("official_raw_reprojection", "history_official_raw", True),
    ("official_clean_full_16", "history_official_clean", False),
    ("full_raw_reprojection", "history_full_raw", True),
)


def summarize(rows: list[dict]) -> dict:
    result = {}
    for variant, _, _ in VARIANTS:
        part = [row for row in rows if row["variant"] == variant]
        tp = sum(row["gt_fire"] and row["fired"] for row in part)
        fp = sum(not row["gt_fire"] and row["fired"] for row in part)
        fn = sum(row["gt_fire"] and not row["fired"] for row in part)
        tn = sum(not row["gt_fire"] and not row["fired"] for row in part)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        routed_true = [row for row in part if row["gt_fire"] and row["fired"]]
        result[variant] = {
            "n": len(part),
            "gate_accuracy": (tp + tn) / len(part),
            "gate_precision": precision,
            "gate_recall": recall,
            "gate_f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "expert_calls": sum(row["fired"] for row in part),
            "expert_call_rate": sum(row["fired"] for row in part) / len(part),
            "gate_reason_accuracy_on_gt_positive": sum(
                row["reason"] == row["gt_reason"] for row in part if row["gt_fire"]
            ) / max(sum(row["gt_fire"] for row in part), 1),
            "expert_type_accuracy_when_true_positive_routed": (
                sum(row["expert_type_correct"] for row in routed_true)
                / max(len(routed_true), 1)
            ),
            "end_to_end_type_accuracy_all": sum(
                row["final_type_correct"] for row in part
            ) / len(part),
            "end_to_end_type_success_on_gt_positive": sum(
                row["gt_fire"] and row["fired"] and row["expert_type_correct"]
                for row in part
            ) / max(sum(row["gt_fire"] for row in part), 1),
            "end_to_end_semantic_complete_on_gt_positive": sum(
                row["gt_fire"] and row["fired"]
                and bool(row["semantic"])
                and row["semantic"]["all_required_concepts"]
                for row in part
            ) / max(sum(row["gt_fire"] for row in part), 1),
            "judger_raw_counts": dict(Counter(row["judger_raw"] for row in part)),
            "expert_pred_type_counts": dict(Counter(
                row["pred_type"] for row in part if row["fired"]
            )),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-jsonl", type=Path, required=True)
    parser.add_argument("--expert-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judger", required=True)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--tau-reproj", type=float, default=0.12)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [json.loads(line) for line in args.gate_jsonl.read_text().splitlines()]
    expert_cases = {
        row["id"]: row
        for row in map(json.loads, args.expert_jsonl.read_text().splitlines())
    }
    if len(cases) != 25 or len(expert_cases) != 14:
        raise ValueError("Expected 25 gate and 14 positive expert cases")

    judger = QwenVL(args.judger, max_new_tokens=24)
    rows = []
    for case in cases:
        original = load_indices(case["video"], case["frame_indices"])
        kept, keep_local = reproject_keep(original, tau=args.tau_reproj)
        for variant, history_key, use_reprojection in VARIANTS:
            frames = kept if use_reprojection else original
            history = case[history_key]
            gate_text = judger_prompt(
                case["protocol"], case["asr_text"],
                proactive_next_step=True, current_step=case["current_step"],
            )
            gate_raw, gate_latency = judger.generate(
                frames, gate_text, history=history
            )
            fired, reason = parse_judger(gate_raw)
            row = {
                "id": case["id"], "variant": variant,
                "gt_fire": case["gt_fire"], "gt_reason": case["gt_reason"],
                "gt_type": case["gt_type"], "fired": fired, "reason": reason,
                "judger_raw": gate_raw,
                "current_step_sent_to_both": case["current_step"],
                "history_sent_to_both": history,
                "judger_prompt": gate_text,
                "n_frames_original": len(original),
                "n_frames_sent": len(frames), "keep_local": keep_local,
                "latency_judger_s": gate_latency,
                "_history_key": history_key,
                "_use_reprojection": use_reprojection,
            }
            rows.append(row)
            print("gate", case["id"], variant, repr(gate_raw), flush=True)

    # Save the complete routing evidence before replacing the 3B model in memory.
    (args.out_dir / "gate_predictions.jsonl").write_text("".join(
        json.dumps({k: v for k, v in row.items() if not k.startswith("_")},
                   ensure_ascii=False) + "\n"
        for row in rows
    ))
    del judger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Loading the models in two phases keeps the 3B and 32B weights from occupying
    # one GPU simultaneously. Routing is still conditional on the recorded 3B result.
    expert = QwenVL(args.expert, max_new_tokens=96)
    cases_by_id = {case["id"]: case for case in cases}
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for row in rows:
            case = cases_by_id[row["id"]]
            expert_text = ""
            expert_raw = ""
            pred_type = "NONE"
            message = ""
            expert_latency = 0.0
            if row["fired"]:
                original = load_indices(case["video"], case["frame_indices"])
                frames = (
                    reproject_keep(original, tau=args.tau_reproj)[0]
                    if row["_use_reprojection"] else original
                )
                history = case[row["_history_key"]]
                expert_text = expert_prompt(
                    case["protocol"], case["asr_text"],
                    judger_reason=row["reason"] or "unknown",
                    current_step=case["current_step"],
                )
                expert_raw, expert_latency = expert.generate(
                    frames, expert_text, history=history
                )
                pred_type, message = parse_expert(expert_raw)

            positive = expert_cases.get(case["id"])
            semantic = (
                score_semantics(message, positive["semantic_rubric"])
                if positive is not None and row["fired"] else None
            )
            row.update({
                "pred_type": pred_type,
                "expert_type_correct": bool(
                    row["fired"] and pred_type.lower() == case["gt_type"]
                ),
                "final_type_correct": pred_type.lower() == case["gt_type"],
                "message": message, "expert_raw": expert_raw,
                "semantic": semantic, "expert_prompt": expert_text,
                "latency_expert_s": expert_latency,
            })
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            output.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            output.flush()
            print("expert", case["id"], row["variant"], pred_type, message, flush=True)

    rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]

    summary = summarize(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "run.json").write_text(json.dumps({
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judger": args.judger, "expert": args.expert,
        "tau_reproj": args.tau_reproj, "gate_jsonl": str(args.gate_jsonl),
        "expert_jsonl_for_scoring_only": str(args.expert_jsonl),
        "variants": [variant for variant, _, _ in VARIANTS],
        "n_gate_calls": len(rows),
        "n_expert_calls": sum(row["fired"] for row in rows),
        "routing": "32B called only after parsed positive 3B decision; predicted reason forwarded",
        "model_context": "same dataset current_step and selected cumulative history sent to both models",
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
