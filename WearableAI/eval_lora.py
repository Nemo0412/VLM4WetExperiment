#!/usr/bin/env python3
"""Generate proactive predictions with a LoRA adapter (official protocol).

Writes predictions.jsonl compatible with starter_kit/run_evaluation.py --eval-only.

Usage:
  python eval_lora.py \\
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
    --adapter /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/best \\
    --golden /scratch/ll5914/datasets/wearable-ai/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \\
    --video-folder /scratch/ll5914/datasets/wearable-ai/egoproactive/val \\
    --predictions /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from proactive_protocol import build_messages, extract_cumulative_frames, video_file


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_mm_messages(frames: list[object], messages: list[dict[str, str]]) -> list[dict]:
    mm: list[dict] = []
    images_inserted = False
    for msg in messages:
        role, text = msg["role"], msg["content"]
        if role == "user" and not images_inserted and frames:
            content: list[dict] = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": text})
            mm.append({"role": role, "content": content})
            images_inserted = True
        else:
            mm.append({"role": role, "content": text})
    return mm


@torch.inference_mode()
def generate_one(
    model,
    processor,
    frames: list[object],
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    mm = to_mm_messages(frames, messages)
    text = processor.apply_chat_template(mm, tokenize=False, add_generation_prompt=True)
    if frames:
        inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--frames-per-interval", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    rows = load_jsonl(args.golden)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.adapter)
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    os.makedirs(os.path.dirname(args.predictions) or ".", exist_ok=True)
    with open(args.predictions, "w") as out_f:
        for row in tqdm(rows, desc="eval_lora"):
            intervals = row["video_intervals"]
            num_chunks = len(intervals)
            vp = video_file(args.video_folder, row["video_path"])
            query = str(row.get("query", ""))
            dialog = row.get("dialog", [])

            answers: list[str] = []
            for j in range(num_chunks):
                frames = extract_cumulative_frames(
                    vp,
                    intervals,
                    j,
                    frames_per_interval=args.frames_per_interval,
                    max_frames=args.max_frames,
                )
                dialog_at_chunk = dialog[j] if j < len(dialog) else []
                messages = build_messages(
                    query,
                    dialog_at_chunk,
                    max_history_turns=args.max_history_turns,
                )
                answers.append(
                    generate_one(model, processor, frames, messages, args.max_new_tokens)
                )

            out_f.write(
                json.dumps({"video_path": row["video_path"], "answers": answers}) + "\n"
            )

    print(f"[eval_lora] wrote {args.predictions}")


if __name__ == "__main__":
    main()
