#!/usr/bin/env python3
"""Measure LabGate four-way accuracy and gated-vs-always-expert latency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from models import QwenVL
from pipeline import LabGate, load_indices


def norm_type(value: str | None) -> str:
    return (value or "NONE").strip().lower()


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    preds = [norm_type(row["pred_type"]) for row in rows]
    gts = [norm_type(row["gt_type"]) for row in rows]
    tp = sum(row["gt_fire"] and row["fired"] for row in rows)
    fp = sum(not row["gt_fire"] and row["fired"] for row in rows)
    fn = sum(row["gt_fire"] and not row["fired"] for row in rows)
    tn = sum(not row["gt_fire"] and not row["fired"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    per_type = defaultdict(lambda: {"n": 0, "type_ok": 0, "gate_ok": 0})
    for row, pred, gt in zip(rows, preds, gts):
        per_type[gt]["n"] += 1
        per_type[gt]["type_ok"] += int(pred == gt)
        per_type[gt]["gate_ok"] += int(row["fired"] == row["gt_fire"])

    always_latency = [
        row["latency_always_s"] for row in rows if "latency_always_s" in row
    ]
    gated_latency = [row["latency_e2e_s"] for row in rows]
    always_accuracy = None
    if any("always_type" in row for row in rows):
        always_accuracy = sum(
            norm_type(row.get("always_type")) == norm_type(row["gt_type"])
            for row in rows
        ) / max(n, 1)

    return {
        "n": n,
        "e2e_type_accuracy": sum(p == g for p, g in zip(preds, gts)) / max(n, 1),
        "always_expert_type_accuracy": always_accuracy,
        "gate_accuracy": (tp + tn) / max(n, 1),
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "gate_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "fire_rate": mean(row["fired"] for row in rows),
        "per_type": {
            key: {
                "n": value["n"],
                "type_acc": value["type_ok"] / value["n"],
                "gate_acc": value["gate_ok"] / value["n"],
            }
            for key, value in per_type.items()
        },
        "confusion_gt_pred": {
            f"{gt}->{pred}": count
            for (gt, pred), count in Counter(zip(gts, preds)).items()
        },
        "mean_frames_in": mean(row["n_frames_in"] for row in rows),
        "mean_frames_kept": mean(row["n_frames_kept"] for row in rows),
        "latency_s": {
            "judger_mean": mean(row["latency_judger_s"] for row in rows),
            "expert_mean_when_fired": mean(
                row["latency_expert_s"] for row in rows if row["fired"]
            ),
            "e2e_gated_mean": mean(gated_latency),
            "e2e_always_expert_mean": (
                mean(always_latency) if always_latency else None
            ),
            "speedup_vs_always": (
                mean(always_latency) / max(mean(gated_latency), 1e-9)
                if always_latency
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judger", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--expert", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--tau-reproj", type=float, default=0.12)
    parser.add_argument("--always-expert", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    clips = [
        json.loads(line)
        for line in Path(args.eval_jsonl).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        clips = clips[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = out_path.with_suffix(".jsonl")
    completed = {}
    if prediction_path.exists():
        for line in prediction_path.read_text().splitlines():
            row = json.loads(line)
            completed[row["id"]] = row

    judger = QwenVL(args.judger, max_new_tokens=24)
    expert = QwenVL(args.expert, max_new_tokens=96)
    gate = LabGate(judger, expert, tau_reproj=args.tau_reproj)
    rows = list(completed.values())

    with prediction_path.open("a") as output:
        for clip in clips:
            if clip["id"] in completed:
                continue
            frames = load_indices(clip["video"], clip["frame_indices"])
            result = gate.run_frames(
                frames, clip["protocol"], clip.get("asr_text") or ""
            )
            row = {
                "id": clip["id"],
                "gt_type": clip["gt_type"],
                "gt_fire": clip["gt_fire"],
                "pred_type": result.output_type,
                "fired": result.fired,
                "judger_reason": result.judger_reason,
                "judger_raw": result.judger_raw,
                "expert_raw": result.expert_raw,
                "message": result.message,
                "n_frames_in": result.n_frames_in,
                "n_frames_kept": result.n_frames_kept,
                "latency_judger_s": result.latency_judger_s,
                "latency_expert_s": result.latency_expert_s,
                "latency_e2e_s": result.latency_e2e_s,
                "note": clip.get("note"),
            }
            if args.always_expert:
                baseline = gate.run_frames(
                    frames,
                    clip["protocol"],
                    clip.get("asr_text") or "",
                    always_expert=True,
                )
                row["always_type"] = baseline.output_type
                row["latency_always_s"] = baseline.latency_expert_s
            rows.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()

    summary = summarize(rows)
    summary.update(
        {"judger": args.judger, "expert": args.expert, "tau_reproj": args.tau_reproj}
    )
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
