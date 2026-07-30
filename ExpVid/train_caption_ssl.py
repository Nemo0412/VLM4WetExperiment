#!/usr/bin/env python3
"""LoRA SSL on ExpVid image–caption pairs: next-token prediction of captions.

Given an image, teacher-force the asr_caption tokens (causal LM loss).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


def find_lora_targets(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                names.add(leaf)
    return sorted(names) or ["q_proj", "v_proj"]


class CaptionDataset(Dataset):
    def __init__(self, jsonl: str, root: str):
        self.root = Path(root)
        self.rows = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        print(f"[data] {jsonl}: {len(self.rows)} pairs", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image_path"]).convert("RGB")
        return {"image": img, "caption": r["asr_caption"], "id": r.get("video_path", str(i))}


def collate(batch, processor):
    assert len(batch) == 1
    b0 = batch[0]
    image = b0["image"]
    caption = b0["caption"]
    user = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this experimental video frame in one sentence."},
        ],
    }]
    full = user + [{"role": "assistant", "content": caption}]
    text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], padding=True)
    for k, v in list(inputs.items()):
        if not torch.is_tensor(v):
            inputs[k] = torch.as_tensor(v)

    prompt_text = processor.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True
    )
    prompt = processor(text=[prompt_text], images=[image], padding=True)
    prompt_ids = prompt["input_ids"]
    if not torch.is_tensor(prompt_ids):
        prompt_ids = torch.as_tensor(prompt_ids)
    labels = inputs["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100
    inputs["labels"] = labels
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ll5914/Labos/ExpVid")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    configure_processor_pixels(processor, args.max_pixels, args.min_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    targets = find_lora_targets(model)
    print(f"[lora] {targets} r={args.lora_r}", flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ),
    )
    for n, p in model.named_parameters():
        if "visual" in n:
            p.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    ds = CaptionDataset(args.train_jsonl, args.root)

    def _collate(batch):
        return collate(batch, processor)

    dl_kwargs = dict(
        batch_size=1, shuffle=True, num_workers=args.num_workers,
        collate_fn=_collate, pin_memory=False, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    dl = DataLoader(ds, **dl_kwargs)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"[train] SSL caption NTP steps={total_steps}", flush=True)

    logf = open(out / "train_log.jsonl", "a")
    gstep = micro = 0
    optim.zero_grad(set_to_none=True)
    for epoch in range(math.ceil(args.epochs)):
        for batch in dl:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out_m = model(**batch)
            loss = out_m.loss
            if loss is None or not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                optim.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    msg = {"step": gstep, "loss": float(loss), "mem_gb": round(mem, 2)}
                    print(f"[step {gstep}/{total_steps}] loss={loss:.4f} mem={mem:.1f}GB", flush=True)
                    logf.write(json.dumps(msg) + "\n")
                    logf.flush()
                if gstep % args.save_every == 0 or gstep >= total_steps:
                    ckpt = out / f"checkpoint-{gstep}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt))
                    processor.save_pretrained(str(ckpt))
                    (ckpt / "config_task.json").write_text(json.dumps({
                        "task": "expvid_image_caption_ssl",
                        "model_path": args.model_path,
                        "global_step": gstep,
                    }, indent=2))
                    print(f"[ckpt] {ckpt}", flush=True)
                if gstep >= total_steps:
                    break
        if gstep >= total_steps:
            break

    model.save_pretrained(str(out))
    processor.save_pretrained(str(out))
    (out / "TRAINING_DONE").write_text(f"step={gstep}\n")
    logf.close()
    print(f"[done] {out} step={gstep}", flush=True)


if __name__ == "__main__":
    main()
