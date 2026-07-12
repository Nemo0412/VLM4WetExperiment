#!/usr/bin/env python3
"""Compare streaming HALT on FineBio correct vs mistake video pairs.

Loads the LoRA checkpoint once, runs chunked streaming on each video,
scores CONTINUE on correct / HALT on mistake (max 6 for 3 pairs).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from peft import PeftModel
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from infer_stream import (
    PROTOCOL_NAMES,
    build_overwrite_config,
    build_prompt,
    load_chunk,
    parse_text_halt,
    read_step_spans,
)

CASES = [
    {
        "name": "P06_03",
        "proto": 3,
        "note": "missing sterile water wash",
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P06_03_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos/P06_03_02.mp4",
        "ann_correct": "/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations/P06_03_01.txt",
        "ann_mistake": None,
    },
    {
        "name": "P17_02",
        "proto": 2,
        "note": "extra PBS wash",
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P17_02_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos/P17_02_02.mp4",
        "ann_correct": "/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations/P17_02_01.txt",
        "ann_mistake": None,
    },
    {
        "name": "P11_06",
        "proto": 6,
        "note": "extra wash buffer",
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P10_06_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos/P11_06_01.mp4",
        "ann_correct": "/scratch/ll5914/Labos/Llava/data/FineBio/action_annotations/P10_06_01.txt",
        "ann_mistake": None,
    },
]


def build_chunks(video: str, ann: str | None, steps_per_chunk: int, window_sec: float):
    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps() or 30.0)
    duration = len(vr) / fps
    chunks = []
    if ann and Path(ann).exists():
        spans = read_step_spans(Path(ann))
        i = 0
        while i < len(spans):
            g = spans[i: i + steps_per_chunk]
            chunks.append({"t0": g[0][0], "t1": g[-1][1]})
            i += steps_per_chunk
    else:
        t = 0.0
        while t < duration:
            chunks.append({"t0": t, "t1": min(t + window_sec, duration)})
            t += window_sec
    return chunks, fps


def run_stream(model, tokenizer, image_processor, halt_head, video, ann, protocol_id,
               num_frames, steps_per_chunk, window_sec, halt_threshold, conv_mode):
    chunks, fps = build_chunks(video, ann, steps_per_chunk, window_sec)
    history = []
    device = model.device
    with torch.inference_mode():
        for ci, ch in enumerate(chunks):
            video_t = load_chunk(video, ch["t0"], ch["t1"], fps, num_frames, image_processor)
            video_t = video_t.to(device, dtype=torch.bfloat16)
            prompt, conv = build_prompt(protocol_id, conv_mode)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            attn = input_ids.ne(tokenizer.pad_token_id or 0).long()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            p_halt = None
            pred_cls = None
            if halt_head is not None:
                out = model(
                    input_ids=input_ids, attention_mask=attn,
                    images=[video_t], modalities=["video"],
                    output_hidden_states=True, return_dict=True,
                )
                z = out.hidden_states[-1][:, -1, :].to(torch.bfloat16)
                probs = F.softmax(halt_head(z).float(), dim=-1)[0]
                p_halt = float(1.0 - probs[0].item())
                pred_cls = int(probs.argmax().item())

            output_ids = model.generate(
                inputs=input_ids, images=[video_t], attention_mask=attn,
                modalities="video", do_sample=False, temperature=1e-5,
                max_new_tokens=128, use_cache=True, stopping_criteria=[stopping],
            )
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            is_halt, _ = parse_text_halt(text)
            if p_halt is not None and p_halt >= halt_threshold:
                is_halt = True

            history.append({
                "chunk": ci, "t0": ch["t0"], "t1": ch["t1"],
                "text": text, "p_halt": p_halt, "pred_cls": pred_cls, "halted": is_halt,
            })
            if is_halt:
                return {
                    "verdict": "HALT",
                    "halt_chunk": ci,
                    "t0": ch["t0"],
                    "t1": ch["t1"],
                    "reason": text,
                    "p_halt": p_halt,
                    "n_chunks": len(chunks),
                    "trace": history,
                }
    return {
        "verdict": "CONTINUE",
        "halt_chunk": None,
        "reason": history[-1]["text"] if history else "CONTINUE",
        "p_halt": history[-1]["p_halt"] if history else None,
        "n_chunks": len(chunks),
        "trace": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/scratch/ll5914/Labos/FineBioStreaming/outputs/streaming_vlm_v1")
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/FineBioStreaming/outputs/eval_correct_mistake")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=2)
    ap.add_argument("--steps-per-chunk", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--halt-threshold", type=float, default=0.45)
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    ckpt = Path(args.checkpoint)
    scfg = {}
    if (ckpt / "config_streaming.json").exists():
        scfg = json.loads((ckpt / "config_streaming.json").read_text())
    base = scfg.get("model_path", args.model_path)
    num_frames = scfg.get("num_frames", args.num_frames)
    pool_stride = scfg.get("pool_stride", args.pool_stride)

    overwrite = build_overwrite_config(base, num_frames, pool_stride)
    model_name = get_model_name_from_path(base)
    print(f"[eval] loading {base} + {ckpt}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base, None, model_name, torch_dtype="bfloat16",
        overwrite_config=overwrite, attn_implementation=args.attn_impl,
    )
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt))
    halt_head = None
    aux = ckpt / "aux_heads.bin"
    if aux.exists():
        blob = torch.load(aux, map_location="cpu", weights_only=False)
        halt_head = nn.Linear(model.config.hidden_size, blob["n_halt"])
        halt_head.load_state_dict(blob["halt_head"])
        halt_head = halt_head.to(model.device, dtype=torch.bfloat16).eval()
    model.eval()

    rows = []
    total = 0
    for case in CASES:
        print("=" * 70, flush=True)
        print(f"[case] {case['name']} proto={case['proto']} ({case['note']})", flush=True)
        for split, video, ann, expect in (
            ("correct", case["correct"], case["ann_correct"], "CONTINUE"),
            ("mistake", case["mistake"], case["ann_mistake"], "HALT"),
        ):
            print(f"\n--- {case['name']} / {split} ---", flush=True)
            res = run_stream(
                model, tokenizer, image_processor, halt_head,
                video, ann, case["proto"],
                num_frames, args.steps_per_chunk, args.window_sec,
                args.halt_threshold, args.conv_mode,
            )
            ok = res["verdict"] == expect
            pts = 1 if ok else 0
            total += pts
            row = {
                "case": case["name"],
                "split": split,
                "expect": expect,
                "got": res["verdict"],
                "correct": ok,
                "points": pts,
                "halt_chunk": res.get("halt_chunk"),
                "t0": res.get("t0"),
                "t1": res.get("t1"),
                "p_halt": res.get("p_halt"),
                "n_chunks": res.get("n_chunks"),
                "reason": res.get("reason", "")[:300],
                "note": case["note"],
            }
            rows.append(row)
            (out_dir / f"{case['name']}_{split}.json").write_text(json.dumps(res, indent=2))
            print(
                f"[{split}] expect={expect} got={res['verdict']} "
                f"pts={pts} halt_chunk={res.get('halt_chunk')} "
                f"t={res.get('t0')}-{res.get('t1')} p_halt={res.get('p_halt')}",
                flush=True,
            )
            print(f"  reason: {str(res.get('reason', ''))[:200]}", flush=True)

    report = {
        "checkpoint": str(ckpt),
        "halt_threshold": args.halt_threshold,
        "total_points": total,
        "max_points": 6,
        "accuracy": total / 6,
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = ["# FineBioStreaming correct vs mistake eval", ""]
    md.append(f"**Checkpoint:** `{ckpt}`  ")
    md.append(f"**Score:** **{total}/6** ({100 * total / 6:.0f}%)  ")
    md.append(f"**halt_threshold:** {args.halt_threshold}")
    md.append("")
    md.append("| Case | Split | Expect | Got | OK | Halt chunk | t (s) | P(halt) |")
    md.append("|------|-------|--------|-----|----|------------|-------|---------|")
    for r in rows:
        tspan = f"{r['t0']:.1f}-{r['t1']:.1f}" if r.get("t0") is not None else "-"
        md.append(
            f"| {r['case']} | {r['split']} | {r['expect']} | {r['got']} | "
            f"{'Y' if r['correct'] else 'N'} | {r['halt_chunk']} | {tspan} | "
            f"{r['p_halt'] if r['p_halt'] is not None else '-'} |"
        )
    md.append("")
    md.append("## Reasons")
    for r in rows:
        md.append(f"### {r['case']} / {r['split']}")
        md.append(f"- note: {r['note']}")
        md.append(f"- `{r['reason']}`")
        md.append("")
    (out_dir / "report.md").write_text("\n".join(md))
    print("=" * 70, flush=True)
    print(f"[done] {total}/6 -> {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
