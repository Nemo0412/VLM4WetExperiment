#!/usr/bin/env python3
"""Convert EgoProactive JSONL into SFT JSONL (official proactive protocol).

Each output line is one chunk-level decision. Train/val split is by video_id
to avoid leakage. Gold answers enter the training target only — not as leaked
future context beyond the chunk's dialog history.

Usage:
  python prepare_sft_data.py \\
    --golden /scratch/ll5914/datasets/wearable-ai/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \\
    --out-dir /scratch/ll5914/Labos/WearableAI/data/sft
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from proactive_protocol import parse_decision


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.golden)
    rng = random.Random(args.seed)

    samples: list[dict] = []
    for row in rows:
        video_path = row["video_path"]
        video_id = Path(video_path).stem
        intervals = row["video_intervals"]
        answers = row["answers"]
        dialog = row.get("dialog", [])
        query = str(row.get("query", ""))

        for chunk_index, answer in enumerate(answers):
            decision, _ = parse_decision(str(answer))
            if decision == "Unknown":
                continue
            dialog_at_chunk = dialog[chunk_index] if chunk_index < len(dialog) else []
            samples.append(
                {
                    "video_path": video_path,
                    "video_id": video_id,
                    "chunk_index": chunk_index,
                    "video_intervals": intervals,
                    "query": query,
                    "dialog_at_chunk": dialog_at_chunk,
                    "target": str(answer).strip(),
                    "decision": decision,
                    "domain": row.get("domain", ""),
                    "task": row.get("task", ""),
                }
            )

    video_ids = sorted({s["video_id"] for s in samples})
    rng.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * args.val_ratio))
    val_ids = set(video_ids[:n_val])

    train_samples = [s for s in samples if s["video_id"] not in val_ids]
    val_samples = [s for s in samples if s["video_id"] in val_ids]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, subset in [("train", train_samples), ("val", val_samples)]:
        out_path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        n_int = sum(1 for s in subset if s["decision"] == "Interrupt")
        print(
            f"[prepare] {name}: videos={len({s['video_id'] for s in subset})} "
            f"samples={len(subset)} interrupt={n_int} "
            f"({n_int / max(len(subset), 1):.1%}) -> {out_path}"
        )


if __name__ == "__main__":
    main()
