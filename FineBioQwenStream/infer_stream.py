#!/usr/bin/env python3
"""Infer CONTINUE/HALT for a FineBio prefix video window with Qwen2.5-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from decord import VideoReader, cpu
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from protocol_prompt import PROTOCOL_NAMES, build_user_prompt, parse_decision


def load_clip(video: str, t0: float, t1: float, nframes: int):
    vr = VideoReader(video, ctx=cpu(0), num_threads=2)
    fps = float(vr.get_avg_fps() or 30.0)
    n = len(vr)
    if t1 is None or t1 <= t0:
        idxs = [int(i) for i in __import__("numpy").linspace(0, max(n - 1, 0), nframes)]
    else:
        times = [t0 + i * (t1 - t0) / max(nframes - 1, 1) for i in range(nframes)]
        idxs = [min(max(int(round(t * fps)), 0), n - 1) for t in times]
    return vr.get_batch(idxs).asnumpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--protocol", type=int, required=True)
    ap.add_argument("--steps", nargs="+", required=True, help="Full ordered step names")
    ap.add_argument("--prefix-len", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=64)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    scfg = {}
    if (ckpt / "config_task.json").exists():
        scfg = json.loads((ckpt / "config_task.json").read_text())
    base = scfg.get("model_path", args.model_path)
    nframes = int(scfg.get("max_frames", args.num_frames))

    processor = AutoProcessor.from_pretrained(
        str(ckpt) if (ckpt / "preprocessor_config.json").exists() else base,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()

    frames = load_clip(args.video, args.t0, args.t1, nframes)
    user = build_user_prompt(args.protocol, args.steps, args.prefix_len)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    # decode only new tokens
    gen = out[:, inputs["input_ids"].shape[1]:]
    reply = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    decision, et = parse_decision(reply)
    print(reply)
    print(f"[parse] decision={decision} error_type={et}")
    print(f"[proto] {args.protocol}: {PROTOCOL_NAMES.get(args.protocol)}")


if __name__ == "__main__":
    main()
