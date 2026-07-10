#!/usr/bin/env python3
"""FineBio LLaVA fine-tuning with LM + compliance classification + protocol aux.

    L_total = L_lm + lambda_comp * L_comp + lambda_proto * L_proto

  L_lm   : standard autoregressive CE on target answer tokens (no token re-weighting)
  L_comp : 2-class CE on pooled multimodal features (NOT_FOLLOWED=0, FOLLOWED=1)
  L_proto: 7-way protocol classification on the same pooled features

Set --lambda-comp 0 --lambda-proto 0 for plain LoRA SFT baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

NOT_FOLLOWED = 0
FOLLOWED = 1


def expand2square(pil_img, background_color):
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        out = Image.new(pil_img.mode, (w, w), background_color)
        out.paste(pil_img, (0, (w - h) // 2))
        return out
    out = Image.new(pil_img.mode, (h, h), background_color)
    out.paste(pil_img, ((h - w) // 2, 0))
    return out


def build_input_ids_labels(conversations, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conv.messages = []
    for j, sent in enumerate(conversations):
        conv.append_message(roles[sent["from"]], sent["value"])
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    target = input_ids.clone()
    tag = conv.roles[1] + ":"
    parts = prompt.split(tag)
    if len(parts) >= 2:
        instr_len = len(tokenizer_image_token(parts[0] + tag, tokenizer))
        target[:instr_len] = IGNORE_INDEX
    return input_ids, target


def compliance_label_from_record(rec: dict) -> int:
    """Scene samples are ignored; compliance comp samples use integrity."""
    if rec["id"].endswith("_scene"):
        return -100
    if "_comp" not in rec["id"]:
        return -100
    return FOLLOWED if int(rec.get("integrity", 0)) == 1 else NOT_FOLLOWED


def load_image(path, image_folder, processor, aspect="pad"):
    img = Image.open(os.path.join(image_folder, path)).convert("RGB")
    if aspect == "pad":
        img = expand2square(img, tuple(int(x * 255) for x in processor.image_mean))
    return processor.preprocess(img, return_tensors="pt")["pixel_values"][0]


class FineBioDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, aspect="pad"):
        self.data = json.load(open(json_path))
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.aspect = aspect

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        pixel = load_image(rec["image"], self.image_folder, self.image_processor, self.aspect)
        input_ids, labels = build_input_ids_labels(rec["conversations"], self.tokenizer)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image": pixel,
            "protocol_id": int(rec.get("protocol_id", -100)),
            "compliance_label": compliance_label_from_record(rec),
        }


@dataclass
class SFTCollator:
    pad_id: int

    def __call__(self, batch):
        ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.pad_id)
        labs = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": ids,
            "labels": labs,
            "attention_mask": ids.ne(self.pad_id),
            "images": torch.stack([b["image"] for b in batch]),
            "protocol_id": torch.tensor([b["protocol_id"] for b in batch], dtype=torch.long),
            "compliance_label": torch.tensor([b["compliance_label"] for b in batch], dtype=torch.long),
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


def masked_mean(hidden, attn_mask):
    m = attn_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conv-version", default="v1")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--proj-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lambda-comp", type=float, default=1.0)
    ap.add_argument("--lambda-proto", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name, device_map=None, torch_dtype=torch.bfloat16,
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

    protocol_head = nn.Linear(hidden, 7).to(device=device, dtype=torch.bfloat16)
    compliance_head = nn.Linear(hidden, 2).to(device=device, dtype=torch.bfloat16)
    model.to(device)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = FineBioDataset(args.train_json, args.image_folder, tokenizer, image_processor)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=SFTCollator(tokenizer.pad_token_id), num_workers=2, drop_last=True,
    )

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

    print(f"[train] samples={len(train_ds)} lambda_comp={args.lambda_comp} "
          f"lambda_proto={args.lambda_proto} steps/epoch={steps_per_epoch} total_steps={total_steps}")
    logf = open(Path(args.output_dir) / "train_log.jsonl", "a")
    gstep = micro = 0
    optim.zero_grad()

    for _epoch in range(math.ceil(args.epochs)):
        for batch in train_dl:
            input_ids = batch["input_ids"][:, :args.max_len].to(device)
            labels = batch["labels"][:, :args.max_len].to(device)
            attn = batch["attention_mask"][:, :args.max_len].to(device)
            images = batch["images"].to(device=device, dtype=torch.bfloat16)
            proto = batch["protocol_id"].to(device)
            comp = batch["compliance_label"].to(device)

            out = model(
                input_ids=input_ids, attention_mask=attn, labels=labels,
                images=images, output_hidden_states=True, return_dict=True,
            )
            lm_loss = out.loss

            z_visual = masked_mean(
                out.hidden_states[-1],
                torch.ones(out.hidden_states[-1].shape[:2], device=device),
            ).to(torch.bfloat16)

            comp_loss = torch.zeros((), device=device)
            if args.lambda_comp > 0:
                compliance_logits = compliance_head(z_visual)  # [B, 2]
                comp_loss = F.cross_entropy(
                    compliance_logits.float(), comp, ignore_index=-100)

            proto_loss = torch.zeros((), device=device)
            if args.lambda_proto > 0:
                proto_loss = F.cross_entropy(
                    protocol_head(z_visual).float(), proto, ignore_index=-100)

            total_loss = lm_loss + args.lambda_comp * comp_loss + args.lambda_proto * proto_loss
            (total_loss / args.grad_accum).backward()
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + proj_params + head_params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                gstep += 1
                if gstep % args.log_every == 0:
                    rec = {
                        "step": gstep, "loss": float(total_loss), "lm": float(lm_loss),
                        "comp": float(comp_loss), "proto": float(proto_loss),
                    }
                    print(f"[step {gstep}/{total_steps}] loss={rec['loss']:.4f} "
                          f"lm={rec['lm']:.4f} comp={rec['comp']:.4f} proto={rec['proto']:.4f}",
                          flush=True)
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
    }, save_dir / "aux_heads.bin")
    logf.close()
    print(f"[done] saved to {save_dir}")


if __name__ == "__main__":
    main()
