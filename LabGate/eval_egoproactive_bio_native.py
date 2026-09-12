#!/usr/bin/env python3
"""Evaluate EgoProactive-Bio through the native LabGate two-stage cascade."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

from eval_zeroshot import summarize
from models import QwenVL
from pipeline import LabGate, load_indices


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judger", required=True)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--tau-reproj", type=float, default=0.12)
    parser.add_argument("--always-expert", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = [json.loads(line) for line in args.eval_jsonl.read_text().splitlines() if line.strip()]
    if len(clips) != 25 or len({clip["id"] for clip in clips}) != 25:
        raise ValueError("Expected 25 unique adapted Bio records")
    model_inputs = {"video", "frame_indices", "protocol", "asr_text"}
    metadata = {
        "status": "loading",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "eval_sha256": hashlib.sha256(args.eval_jsonl.read_bytes()).hexdigest(),
        "model_input_fields": sorted(model_inputs),
        "audit_fields_not_sent_to_models": sorted(set(clips[0]) - model_inputs),
        "pipeline": "native LabGate: load_indices -> reproject_keep -> YES/NO judger -> conditional TYPE/MSG expert",
        "extension": "proactive_next_step=True adds one gate rule and one reason enum; schemas otherwise unchanged",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "numpy", "decord", "accelerate"]
        },
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ["eval_egoproactive_bio_native.py", "pipeline.py", "prompts.py", "models.py"]
        },
    }
    write_json(args.out_dir / "run.json", metadata)
    load_start = time.perf_counter()
    judger = QwenVL(args.judger, max_new_tokens=24)
    expert = QwenVL(args.expert, max_new_tokens=96)
    gate = LabGate(
        judger, expert, tau_reproj=args.tau_reproj, proactive_next_step=True
    )
    metadata["model_load_wall_s"] = time.perf_counter() - load_start
    metadata["device_maps"] = {
        "judger": {key: str(value) for key, value in judger.model.hf_device_map.items()},
        "expert": {key: str(value) for key, value in expert.model.hf_device_map.items()},
    }
    metadata["status"] = "running"
    write_json(args.out_dir / "run.json", metadata)
    rows = []
    with (args.out_dir / "predictions.jsonl").open("w") as output:
        for index, clip in enumerate(clips):
            wall_start = time.perf_counter()
            frames = load_indices(clip["video"], clip["frame_indices"])
            result = gate.run_frames(frames, clip["protocol"], clip["asr_text"])
            row = {
                **{key: value for key, value in clip.items() if key not in model_inputs},
                "fired": result.fired,
                "pred_type": result.output_type,
                "judger_reason": result.judger_reason,
                "judger_raw": result.judger_raw,
                "expert_raw": result.expert_raw,
                "message": result.message,
                "judger_prompt": result.judger_prompt,
                "expert_prompt": result.expert_prompt,
                "n_frames_in": result.n_frames_in,
                "n_frames_kept": result.n_frames_kept,
                "keep_local": result.keep_local,
                "latency_judger_s": result.latency_judger_s,
                "latency_expert_s": result.latency_expert_s,
                "latency_e2e_s": result.latency_e2e_s,
                "latency_wall_s": time.perf_counter() - wall_start,
            }
            if args.always_expert:
                baseline = gate.run_frames(
                    frames, clip["protocol"], clip["asr_text"], always_expert=True
                )
                row.update(
                    always_type=baseline.output_type,
                    always_message=baseline.message,
                    always_raw=baseline.expert_raw,
                    always_expert_prompt=baseline.expert_prompt,
                    latency_always_s=baseline.latency_expert_s,
                )
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            native = summarize(rows)
            extra = {
                "reason_accuracy_on_gt_fire": sum(
                    row["gt_fire"] and row["judger_reason"] == row["gt_reason"] for row in rows
                ) / max(sum(row["gt_fire"] for row in rows), 1),
                "reason_confusion_gt_pred": dict(
                    Counter(f"{row['gt_reason']}->{row['judger_reason'] or 'none'}" for row in rows)
                ),
            }
            write_json(args.out_dir / "summary.json", {"complete": len(rows) == len(clips), **native, **extra})
            print(
                f"{index + 1}/25 {clip['id']} gt={clip['gt_type']} "
                f"gate={result.judger_raw!r} type={result.output_type} msg={result.message}",
                flush=True,
            )
    metadata["status"] = "complete"
    metadata["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.out_dir / "run.json", metadata)


if __name__ == "__main__":
    main()
