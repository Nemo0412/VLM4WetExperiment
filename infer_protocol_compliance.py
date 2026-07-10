#!/usr/bin/env python3
"""Check whether a lab video follows a protocol using LLaVA-NeXT-Video-7B.

Uniformly samples frames from the mp4 and feeds them as native video tokens
(no frame-grid stitching).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoConfig

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_video_tensor(video_path: str, image_processor, num_frames: int) -> torch.Tensor:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]


def build_overwrite_config(model_path: str, num_frames: int, pool_stride: int) -> dict:
    cfg = AutoConfig.from_pretrained(model_path)
    overwrite = {
        "mm_spatial_pool_mode": "average",
        "mm_spatial_pool_stride": pool_stride,
        "mm_newline_position": "grid",
    }
    least = num_frames * (24 // pool_stride) ** 2 + 1000
    scaling = math.ceil(least / 4096)
    if scaling >= 2 and "vicuna" in cfg._name_or_path.lower():
        overwrite["rope_scaling"] = {"factor": float(scaling), "type": "linear"}
        overwrite["max_sequence_length"] = 4096 * scaling
        overwrite["tokenizer_model_max_length"] = 4096 * scaling
    return overwrite


def build_prompt(protocol_text: str, conv_mode: str) -> str:
    question = (
        "These frames are uniformly sampled in temporal order from a laboratory "
        "experiment video.\n\n"
        "Reference protocol:\n"
        f"{protocol_text.strip()}\n\n"
        "Based only on the visible frames, decide whether the experimenter followed this protocol.\n"
        "Answer with:\n"
        "1) Verdict: FOLLOWED or NOT FOLLOWED\n"
        "2) Brief evidence from the frames (what steps you can / cannot observe)\n"
        "3) Any likely deviations or missing steps, if applicable"
    )
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--video", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--pool-stride", type=int, default=2)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    protocol_text = Path(args.protocol).read_text(encoding="utf-8")
    disable_torch_init()

    overwrite = build_overwrite_config(args.model_path, args.num_frames, args.pool_stride)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        torch_dtype="bfloat16", overwrite_config=overwrite, attn_implementation=args.attn_impl,
    )

    video_tensor = load_video_tensor(args.video, image_processor, args.num_frames)
    video_tensor = video_tensor.to(model.device, dtype=torch.bfloat16)
    videos = [video_tensor]

    prompt = build_prompt(protocol_text, args.conv_mode)
    print(f"[INFO] model={args.model_path} frames={args.num_frames} conv={args.conv_mode}")
    print(f"[INFO] video={args.video}")
    print(f"[INFO] protocol={args.protocol}")
    print("=" * 60)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id or 0).long()

    conv = conv_templates[args.conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=videos,
            attention_mask=attention_mask,
            modalities="video",
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[-1].strip()

    print(answer)
    print("=" * 60)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(answer + "\n", encoding="utf-8")
        print(f"[INFO] Saved answer to {args.output}")


if __name__ == "__main__":
    main()
