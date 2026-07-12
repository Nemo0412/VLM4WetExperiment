#!/usr/bin/env python3
"""Streaming inference with LLaVA-NeXT-Video VLM.

Consumes protocol + successive video chunks. Stops on HALT (aux head or text).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

PROTOCOL_NAMES = {
    1: "Cell lysate collection (single PBS wash)",
    2: "Cell lysate collection (double PBS wash)",
    3: "Magnetic-bead DNA extraction (single ethanol wash)",
    4: "Magnetic-bead DNA extraction (double ethanol wash)",
    5: "PCR reaction setup with 8-tube strips",
    6: "Spin-column DNA extraction (two wash steps)",
    7: "Spin-column DNA extraction (three wash steps)",
}
ID2HALT = {
    0: "continue", 1: "missing_step", 2: "redundant_step",
    3: "wrong_order", 4: "within_step_error",
}


def build_overwrite_config(model_path: str, num_frames: int, pool_stride: int) -> dict:
    cfg = AutoConfig.from_pretrained(model_path)
    overwrite = {
        "mm_spatial_pool_mode": "average",
        "mm_spatial_pool_stride": pool_stride,
        "mm_newline_position": "grid",
    }
    least = num_frames * (24 // pool_stride) ** 2 + 1000
    scaling = math.ceil(least / 4096)
    if scaling >= 2 and "vicuna" in getattr(cfg, "_name_or_path", "").lower():
        overwrite["rope_scaling"] = {"factor": float(scaling), "type": "linear"}
        overwrite["max_sequence_length"] = 4096 * scaling
        overwrite["tokenizer_model_max_length"] = 4096 * scaling
    return overwrite


def read_step_spans(ann_path: Path):
    segs = []
    with open(ann_path, newline="") as f:
        for row in csv.DictReader(f):
            task = (row.get("task") or "").strip()
            if not task:
                continue
            segs.append((float(row["start_sec"]), float(row["end_sec"]), task))
    segs.sort()
    out = []
    for s, e, t in segs:
        if out and out[-1][2] == t:
            out[-1] = (out[-1][0], e, t)
        else:
            out.append((s, e, t))
    return out


def load_chunk(video, t0, t1, fps, k, image_processor):
    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    n = len(vr)
    times = [t0 + i * (t1 - t0) / max(k - 1, 1) for i in range(k)]
    idxs = [min(max(int(round(t * fps)), 0), n - 1) for t in times]
    frames = vr.get_batch(idxs).asnumpy()
    return image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]


def build_prompt(protocol_id: int, conv_mode: str) -> str:
    name = PROTOCOL_NAMES.get(protocol_id, f"protocol {protocol_id}")
    q = (
        f"You are monitoring a wet-lab experiment in real time. "
        f"The intended protocol is protocol {protocol_id}: {name}. "
        "Based on the latest video chunk, decide whether to CONTINUE or HALT. "
        "If HALT, state the error type and brief reason."
    )
    qs = DEFAULT_IMAGE_TOKEN + "\n" + q
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def parse_text_halt(text: str) -> tuple[bool, str]:
    t = text.upper()
    if "HALT" in t or "NOT FOLLOWED" in t:
        return True, text.strip()
    if t.strip().startswith("CONTINUE"):
        return False, text.strip()
    return "HALT" in text, text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="LoRA dir; if None use base VLM only")
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    ap.add_argument("--video", required=True)
    ap.add_argument("--ann", default=None)
    ap.add_argument("--protocol-id", type=int, required=True)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=2)
    ap.add_argument("--steps-per-chunk", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--halt-threshold", type=float, default=0.45)
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    disable_torch_init()
    base = args.model_path
    cfg_path = Path(args.checkpoint) / "config_streaming.json" if args.checkpoint else None
    if cfg_path and cfg_path.exists():
        scfg = json.loads(cfg_path.read_text())
        base = scfg.get("model_path", base)
        args.num_frames = scfg.get("num_frames", args.num_frames)

    overwrite = build_overwrite_config(base, args.num_frames, args.pool_stride)
    model_name = get_model_name_from_path(base)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base, None, model_name, torch_dtype="bfloat16",
        overwrite_config=overwrite, attn_implementation=args.attn_impl,
    )

    halt_head = None
    if args.checkpoint:
        from peft import PeftModel
        aux = Path(args.checkpoint) / "aux_heads.bin"
        if (Path(args.checkpoint) / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, args.checkpoint)
        if aux.exists():
            blob = torch.load(aux, map_location="cpu")
            halt_head = nn.Linear(model.config.hidden_size, blob["n_halt"])
            halt_head.load_state_dict(blob["halt_head"])
            halt_head = halt_head.to(model.device, dtype=torch.bfloat16).eval()

    model.eval()
    vr = VideoReader(args.video, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps() or 30.0)
    duration = len(vr) / fps

    chunks = []
    if args.ann and Path(args.ann).exists():
        spans = read_step_spans(Path(args.ann))
        i = 0
        while i < len(spans):
            g = spans[i: i + args.steps_per_chunk]
            chunks.append({"t0": g[0][0], "t1": g[-1][1]})
            i += args.steps_per_chunk
    else:
        t = 0.0
        while t < duration:
            chunks.append({"t0": t, "t1": min(t + args.window_sec, duration)})
            t += args.window_sec

    print(f"[stream] VLM={base}")
    print(f"[stream] protocol={args.protocol_id}: {PROTOCOL_NAMES.get(args.protocol_id)}")
    print(f"[stream] chunks={len(chunks)}")
    print("=" * 60)

    history, final = [], None
    with torch.inference_mode():
        for ci, ch in enumerate(chunks):
            video_t = load_chunk(
                args.video, ch["t0"], ch["t1"], fps, args.num_frames, image_processor
            ).to(model.device, dtype=torch.bfloat16)
            prompt, conv = build_prompt(args.protocol_id, args.conv_mode)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            attn = input_ids.ne(tokenizer.pad_token_id or 0).long()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # Aux halt probability (if trained head present)
            p_halt = None
            if halt_head is not None:
                out = model(
                    input_ids=input_ids, attention_mask=attn,
                    images=[video_t], modalities=["video"],
                    output_hidden_states=True, return_dict=True,
                )
                z = out.hidden_states[-1][:, -1, :].to(torch.bfloat16)
                probs = F.softmax(halt_head(z).float(), dim=-1)[0]
                p_halt = float(1.0 - probs[0])
                pred = int(probs.argmax().item())

            output_ids = model.generate(
                inputs=input_ids, images=[video_t], attention_mask=attn,
                modalities="video", do_sample=False, temperature=1e-5,
                max_new_tokens=128, use_cache=True, stopping_criteria=[stopping],
            )
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            is_halt, _ = parse_text_halt(text)
            if p_halt is not None and p_halt >= args.halt_threshold:
                is_halt = True

            line = f"[chunk {ci}] t={ch['t0']:.1f}-{ch['t1']:.1f}s"
            if p_halt is not None:
                line += f" P(halt)={p_halt:.3f}"
            line += f" | {text[:120]}"
            print(line)
            history.append({"chunk": ci, "text": text, "p_halt": p_halt})

            if is_halt:
                print("-" * 60)
                print(text)
                print("[stream] STOP")
                final = {"halt_chunk": ci, "t0": ch["t0"], "t1": ch["t1"],
                         "reason": text, "p_halt": p_halt, "protocol_id": args.protocol_id}
                break
        else:
            final = {"halt_chunk": None, "reason": "CONTINUE through end of stream.",
                     "protocol_id": args.protocol_id}
            print("-" * 60)
            print(final["reason"])
            print("[stream] END")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"final": final, "trace": history}, indent=2))


if __name__ == "__main__":
    main()
