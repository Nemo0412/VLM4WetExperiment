#!/usr/bin/env python3
"""FineBio fine-tuning for LLaVA-NeXT-Video-7B with LM + compliance + protocol aux.

    L_total = L_lm + lambda_comp * L_comp + lambda_proto * L_proto

Fixes vs v1:
  - Official preprocess_v1 label masking (avoids all-IGNORE -> NaN lm loss)
  - Video tuple format (tensor, size, "video") + modalities/image_sizes
  - FP32 LM loss recomputation from logits
  - Default 16 frames (stable for 7B + stride 2)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from packaging import version
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

NOT_FOLLOWED = 0
FOLLOWED = 1


def preprocess_multimodal(sources: Sequence, data_args) -> Sequence:
    if not data_args.is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if (num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"]
                    and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN)):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")
    return sources


def preprocess_v1(sources, tokenizer, has_image: bool = False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations, return_tensors="pt", padding="longest",
            max_length=tokenizer.model_max_length, truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(f"WARNING: tokenization mismatch: {cur_len} vs {total_len} (ignored)")
    return dict(input_ids=input_ids, labels=targets)


def compliance_label_from_record(rec: dict) -> int:
    if rec["id"].endswith("_scene"):
        return -100
    if "_comp" not in rec["id"]:
        return -100
    return FOLLOWED if int(rec.get("integrity", 0)) == 1 else NOT_FOLLOWED


def load_video_tensor(video_path: str, image_processor, num_frames: int,
                      frame_indices: list[int] | None = None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    if frame_indices is None:
        frame_indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist()
    frames = vr.get_batch(frame_indices).asnumpy()
    tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    h, w = int(frames.shape[1]), int(frames.shape[2])
    return tensor, (w, h)


def count_supervised(labels: torch.Tensor) -> int:
    return int((labels != IGNORE_INDEX).sum().item())


class FineBioVideoDataset(Dataset):
    def __init__(self, json_path, video_folder, tokenizer, image_processor, num_frames, data_args):
        self.data = json.load(open(json_path))
        self.video_folder = video_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        video_path = os.path.join(self.video_folder, rec["video"])
        frame_indices = rec.get("frame_indices")
        pixel, size = load_video_tensor(
            video_path, self.image_processor, self.num_frames, frame_indices)

        sources = preprocess_multimodal(
            copy.deepcopy([rec["conversations"]]), self.data_args)
        tok = preprocess_v1(sources, self.tokenizer, has_image=True)
        input_ids = tok["input_ids"][0]
        labels = tok["labels"][0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": [(pixel, size, "video")],
            "protocol_id": int(rec.get("protocol_id", -100)),
            "compliance_label": compliance_label_from_record(rec),
            "n_supervised": count_supervised(labels),
        }


@dataclass
class VideoCollator:
    pad_id: int

    def __call__(self, batch):
        ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_id)
        labs = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX)
        images = [b["image"] for b in batch]
        return {
            "input_ids": ids,
            "labels": labs,
            "attention_mask": ids.ne(self.pad_id),
            "images": [im[0] for im_list in images for im in im_list],
            "image_sizes": [im[1] for im_list in images for im in im_list],
            "modalities": [im[2] for im_list in images for im in im_list],
            "protocol_id": torch.tensor([b["protocol_id"] for b in batch], dtype=torch.long),
            "compliance_label": torch.tensor([b["compliance_label"] for b in batch], dtype=torch.long),
            "n_supervised": sum(b["n_supervised"] for b in batch),
        }


def find_lora_targets(model):
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in ["vision_tower", "mm_projector", "lm_head",
                                    "compliance_head", "protocol_head"]):
            continue
        names.add(name.split(".")[-1])
    return sorted(names)


def answer_pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Pool from last token of the expanded multimodal sequence."""
    return hidden[:, -1, :]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--video-folder", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conv-version", default="vicuna_v1")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--pool-stride", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lambda-comp", type=float, default=1.0)
    ap.add_argument("--lambda-proto", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            f"CUDA unavailable (visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"count={torch.cuda.device_count()})")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    print(f"[cuda] using {device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    data_args = SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False)

    overwrite = build_overwrite_config(args.model_path, args.num_frames, args.pool_stride)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name, device_map="auto", torch_dtype="bfloat16",
        overwrite_config=overwrite, attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_version]

    hidden = model.config.hidden_size
    from peft import LoraConfig, get_peft_model
    targets = find_lora_targets(model)
    print(f"[lora] targets={targets}")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    ))
    base = model.get_base_model()
    for n, p in base.named_parameters():
        if "mm_projector" in n:
            p.requires_grad_(True)
    for p in base.get_vision_tower().parameters():
        p.requires_grad_(False)

    model_device = next(p.device for p in model.parameters() if p.device.type == "cuda")
    protocol_head = nn.Linear(hidden, 7, device=model_device, dtype=torch.bfloat16)
    compliance_head = nn.Linear(hidden, 2, device=model_device, dtype=torch.bfloat16)
    device = model_device
    print(f"[cuda] aux heads on {device}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = FineBioVideoDataset(
        args.train_json, args.video_folder, tokenizer, image_processor,
        args.num_frames, data_args)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=VideoCollator(tokenizer.pad_token_id), num_workers=0, drop_last=True,
    )

    # Sanity-check first sample labels
    s0 = train_ds[0]
    print(f"[sanity] sample0 supervised_tokens={s0['n_supervised']} "
          f"input_len={len(s0['input_ids'])} video_shape={tuple(s0['image'][0][0].shape)}")
    if s0["n_supervised"] == 0:
        raise RuntimeError("First sample has 0 supervised tokens — label mask is broken.")

    lora_params, proj_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (proj_params if "mm_projector" in n else lora_params).append(p)
    head_params = list(protocol_head.parameters()) + list(compliance_head.parameters())
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": proj_params, "lr": args.proj_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (s / warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    print(f"[train] samples={len(train_ds)} frames={args.num_frames} lambda_comp={args.lambda_comp} "
          f"lambda_proto={args.lambda_proto} steps/epoch={steps_per_epoch} total_steps={total_steps}")
    logf = open(Path(args.output_dir) / "train_log.jsonl", "a")
    gstep = micro = 0
    optim.zero_grad()

    for _epoch in range(math.ceil(args.epochs)):
        for batch in train_dl:
            if batch["n_supervised"] == 0:
                print("[warn] skipping batch with 0 supervised tokens")
                continue

            input_ids = batch["input_ids"][:, :args.max_len].to(device)
            labels = batch["labels"][:, :args.max_len].to(device)
            attn = batch["attention_mask"][:, :args.max_len].to(device)
            images = [v.to(device=device, dtype=torch.bfloat16) for v in batch["images"]]
            modalities = batch["modalities"]
            image_sizes = batch["image_sizes"]
            proto = batch["protocol_id"].to(device)
            comp = batch["compliance_label"].to(device)

            out = model(
                input_ids=input_ids, attention_mask=attn,
                labels=labels, images=images, modalities=modalities,
                image_sizes=image_sizes, output_hidden_states=True, return_dict=True,
            )

            lm_loss = out.loss.float() if out.loss is not None else torch.tensor(float("nan"), device=device)
            if not torch.isfinite(lm_loss):
                print(f"[warn] non-finite lm_loss (n_sup={batch['n_supervised']}) — skip batch")
                optim.zero_grad()
                micro += 1
                continue

            z_visual = answer_pool_hidden(out.hidden_states[-1]).to(torch.bfloat16)

            comp_loss = torch.zeros((), device=device)
            if args.lambda_comp > 0:
                comp_loss = F.cross_entropy(
                    compliance_head(z_visual).float(), comp, ignore_index=-100)

            proto_loss = torch.zeros((), device=device)
            if args.lambda_proto > 0:
                proto_loss = F.cross_entropy(
                    protocol_head(z_visual).float(), proto, ignore_index=-100)

            total_loss = lm_loss + args.lambda_comp * comp_loss + args.lambda_proto * proto_loss
            if not torch.isfinite(total_loss):
                print(f"[warn] skip step: total_loss not finite "
                      f"(lm={float(lm_loss)} comp={float(comp_loss)} proto={float(proto_loss)})")
                optim.zero_grad()
                micro += 1
                continue

            (total_loss / args.grad_accum).backward()
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_params + proj_params + head_params, args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                gstep += 1
                if gstep % args.log_every == 0:
                    rec = {
                        "step": gstep, "loss": float(total_loss), "lm": float(lm_loss),
                        "comp": float(comp_loss), "proto": float(proto_loss),
                        "n_sup": batch["n_supervised"],
                    }
                    print(f"[step {gstep}/{total_steps}] loss={rec['loss']:.4f} "
                          f"lm={rec['lm']:.4f} comp={rec['comp']:.4f} proto={rec['proto']:.4f} "
                          f"n_sup={rec['n_sup']}", flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
            if gstep >= total_steps:
                break
        if gstep >= total_steps:
            break

    save_dir = Path(args.output_dir)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    non_lora = {n: p.detach().cpu() for n, p in model.named_parameters()
                if p.requires_grad and "lora_" not in n}
    torch.save(non_lora, save_dir / "non_lora_trainables.bin")
    torch.save({
        "protocol_head": protocol_head.state_dict(),
        "compliance_head": compliance_head.state_dict(),
        "num_frames": args.num_frames,
        "pool_stride": args.pool_stride,
    }, save_dir / "aux_heads.bin")
    logf.close()
    print(f"[done] saved to {save_dir}")


if __name__ == "__main__":
    main()
