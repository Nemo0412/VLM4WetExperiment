"""Qwen2.5-VL load + generate (judger and expert share this wrapper)."""

from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def configure_processor_pixels(processor, max_pixels: int, min_pixels: int):
    for proc in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if proc is None:
            continue
        if hasattr(proc, "max_pixels"):
            proc.max_pixels = max_pixels
        if hasattr(proc, "min_pixels"):
            proc.min_pixels = min_pixels
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            proc.size = {
                **proc.size,
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }


class QwenVL:
    def __init__(
        self,
        model_path: str,
        max_pixels: int = 128 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        max_new_tokens: int = 64,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        configure_processor_pixels(self.processor, max_pixels, min_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, frames: np.ndarray, user_text: str) -> tuple[str, float]:
        if frames.ndim != 4:
            raise ValueError(f"expected THWC frames, got {frames.shape}")
        if len(frames) == 1:
            frames = np.concatenate([frames, frames], axis=0)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], videos=[frames], return_tensors="pt")
        inputs = {
            k: v.to(self.model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[:, inputs["input_ids"].shape[1] :]
        reply = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return reply, dt
