#!/usr/bin/env python3
"""Check whether a lab video follows a given protocol using LLaVA-1.5-7B.

LLaVA is image-only, so we uniformly sample frames and stitch them into a
single grid image before prompting.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def sample_video_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = [int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to read frames from: {video_path}")
    return frames


def make_grid(frames: list[Image.Image], cell_size: int = 336) -> Image.Image:
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(0, 0, 0))
    for i, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        x = (i % cols) * cell_size + (cell_size - thumb.width) // 2
        y = (i // cols) * cell_size + (cell_size - thumb.height) // 2
        grid.paste(thumb, (x, y))
    return grid


def build_prompt(protocol_text: str, model_name: str, use_im_start_end: bool) -> tuple[str, str]:
    question = (
        "These frames are sampled in temporal order from a laboratory experiment video "
        "(left-to-right, top-to-bottom).\n\n"
        "Reference protocol:\n"
        f"{protocol_text.strip()}\n\n"
        "Based only on the visible frames, decide whether the experimenter followed this protocol.\n"
        "Answer with:\n"
        "1) Verdict: FOLLOWED or NOT FOLLOWED\n"
        "2) Brief evidence from the frames (what steps you can / cannot observe)\n"
        "3) Any likely deviations or missing steps, if applicable"
    )

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--video", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--grid-out", default=None, help="Optional path to save frame grid")
    parser.add_argument("--output", default=None, help="Optional path to save model answer")
    args = parser.parse_args()

    protocol_text = Path(args.protocol).read_text(encoding="utf-8")
    frames = sample_video_frames(args.video, num_frames=args.num_frames)
    grid = make_grid(frames)

    if args.grid_out:
        Path(args.grid_out).parent.mkdir(parents=True, exist_ok=True)
        grid.save(args.grid_out)
        print(f"[INFO] Saved frame grid to {args.grid_out}")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    prompt, conv_mode = build_prompt(
        protocol_text,
        model_name,
        use_im_start_end=getattr(model.config, "mm_use_im_start_end", False),
    )
    print(f"[INFO] model={args.model_path} conv_mode={conv_mode} frames={len(frames)}")
    print(f"[INFO] video={args.video}")
    print(f"[INFO] protocol={args.protocol}")
    print("=" * 60)

    images_tensor = process_images([grid], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=[grid.size],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # Some LLaVA builds return the full prompt+answer; keep the assistant turn only.
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
