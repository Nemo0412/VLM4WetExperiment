#!/usr/bin/env python3
"""Qwen2.5-VL LoRA SFT for protocol-level prefix streaming.

Loss (per sample, with early-detection horizon weight w(k)):
  L = w(k) * (L_lm + λ_halt * L_halt + λ_type * L_type)

- L_lm  : teacher-forced CONTINUE / HALT. error_type=... text
- L_halt: binary CE (continue vs halt)
- L_type: 2-way CE on mistake type (HALT only): missing_protocol | wrong_execution
- w(k)  : HALT with k in {1..5} error frames;
         w(k)=1.5/k + 1.0*[k==5]  → prefer 1-frame, must succeed by 5

No SSL. Protocol-level only (concatenated full protocol videos).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from protocol_prompt import ERROR_TYPES


def load_frames(video_path: str, indices: list[int]) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(i, 0), n - 1) for i in indices]
    return vr.get_batch(idxs).asnumpy()


def load_concat_segments(video_root: str, segments: list[dict]) -> np.ndarray:
    """Concatenate frames from multiple full-protocol videos into one array."""
    parts = []
    for seg in segments:
        path = os.path.join(video_root, seg["video"])
        parts.append(load_frames(path, seg["frame_indices"]))
    return np.concatenate(parts, axis=0)


class ProtoPrefixDataset(Dataset):
    def __init__(self, jsonl_path: str, video_root: str, max_frames: int = 128):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.video_root = video_root
        self.max_frames = max_frames
        print(f"[data] {jsonl_path}: {len(self.rows)} samples", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rec = self.rows[i]
        frames = load_concat_segments(self.video_root, rec["segments"])
        if len(frames) > self.max_frames:
            # keep last frames (error evidence is at the end for HALT)
            frames = frames[-self.max_frames:]
        return {
            "frames": frames,
            "user_text": rec["messages"][0]["content"],
            "asst_text": rec["messages"][1]["content"],
            "halt_label": int(rec.get("halt_label", 0 if rec["label"] == "continue" else 1)),
            "type_id": int(rec.get("type_id", -100)),
            "loss_weight": float(rec.get("loss_weight", 1.0)),
            "halt_horizon": rec.get("halt_horizon"),
            "id": rec["id"],
        }


def collate_one(batch, processor):
    assert len(batch) == 1
    b0 = batch[0]
    frames = b0["frames"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": b0["user_text"]},
            ],
        },
        {"role": "assistant", "content": b0["asst_text"]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")

    user_only = [{"role": "user", "content": messages[0]["content"]}]
    prompt_text = processor.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = processor(text=[prompt_text], videos=[frames], return_tensors="pt")["input_ids"]
    labels = inputs["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100
    inputs["labels"] = labels
    inputs["halt_label"] = torch.tensor([b0["halt_label"]], dtype=torch.long)
    inputs["type_id"] = torch.tensor([b0["type_id"]], dtype=torch.long)
    inputs["loss_weight"] = torch.tensor([b0["loss_weight"]], dtype=torch.float32)
    inputs["halt_horizon"] = b0.get("halt_horizon")
    return inputs


def find_lora_targets(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                names.add(leaf)
    return sorted(names) or ["q_proj", "v_proj"]


def pooled_last_hidden(out) -> torch.Tensor:
    """Mean-pool last hidden state → [B, H] for aux heads."""
    h = out.hidden_states[-1]  # [B, T, H]
    return h.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lambda-halt", type=float, default=0.5,
                    help="Weight for explicit CONTINUE/HALT CE")
    ap.add_argument("--lambda-type", type=float, default=1.0,
                    help="Weight for explicit mistake-type CE (HALT only)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    assert args.batch_size == 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load] {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    targets = find_lora_targets(model)
    print(f"[lora] targets={targets}", flush=True)
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

    hidden = model.config.hidden_size
    halt_head = nn.Linear(hidden, 2).to(device=device, dtype=torch.bfloat16)
    type_head = nn.Linear(hidden, len(ERROR_TYPES)).to(device=device, dtype=torch.bfloat16)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    ds = ProtoPrefixDataset(args.train_jsonl, args.video_root, args.max_frames)

    def _collate(batch):
        return collate_one(batch, processor)

    dl = DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
        collate_fn=_collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(halt_head.parameters()) + list(type_head.parameters())
    optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr}],
        weight_decay=0.0,
    )
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (s / warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    print(
        f"[train] protocol-level SFT  L=L_lm+{args.lambda_halt}*L_halt+{args.lambda_type}*L_type  "
        f"steps={total_steps} error_types={list(ERROR_TYPES)}",
        flush=True,
    )
    logf = open(Path(args.output_dir) / "train_log.jsonl", "a")
    gstep = micro = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(math.ceil(args.epochs)):
        for batch in dl:
            halt_y = batch.pop("halt_label").to(device)
            type_y = batch.pop("type_id").to(device)
            w = batch.pop("loss_weight").to(device).float().mean()
            batch.pop("halt_horizon", None)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(**batch, output_hidden_states=True)
            lm_loss = out.loss
            if lm_loss is None or not torch.isfinite(lm_loss):
                print("[warn] bad lm_loss, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            z = pooled_last_hidden(out).to(torch.bfloat16)
            l_halt = F.cross_entropy(halt_head(z).float(), halt_y)
            mask = type_y >= 0
            if mask.any():
                l_type = F.cross_entropy(type_head(z).float()[mask], type_y[mask])
            else:
                l_type = torch.zeros((), device=device)

            # Early detection: L = w(k) * (L_lm + λ_halt L_halt + λ_type L_type)
            # w(1) high (prefer 1-frame), w(5) also high (must detect by 5).
            base = lm_loss + args.lambda_halt * l_halt + args.lambda_type * l_type
            total = w * base
            if not torch.isfinite(total):
                print("[warn] bad total, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            (total / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 0.5)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    msg = {
                        "step": gstep,
                        "loss": float(total),
                        "lm": float(lm_loss),
                        "halt": float(l_halt),
                        "type": float(l_type),
                        "w": float(w),
                        "mem_gb": round(mem, 2),
                    }
                    print(
                        f"[step {gstep}/{total_steps}] loss={total:.4f} "
                        f"lm={lm_loss:.4f} halt={l_halt:.4f} type={l_type:.4f} "
                        f"w={float(w):.2f} mem={mem:.1f}GB",
                        flush=True,
                    )
                    logf.write(json.dumps(msg) + "\n")
                    logf.flush()
                if gstep % args.save_every == 0 or gstep >= total_steps:
                    ckpt = Path(args.output_dir) / f"checkpoint-{gstep}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt))
                    processor.save_pretrained(str(ckpt))
                    torch.save({
                        "halt_head": halt_head.state_dict(),
                        "type_head": type_head.state_dict(),
                        "error_types": list(ERROR_TYPES),
                        "global_step": gstep,
                        "base_model": args.model_path,
                    }, ckpt / "aux_heads.bin")
                    (ckpt / "config_task.json").write_text(json.dumps({
                        "task": "finebio_protocol_prefix_stream_sft",
                        "level": "protocol",
                        "model_path": args.model_path,
                        "global_step": gstep,
                        "max_frames": args.max_frames,
                        "loss": "L_lm + λ_halt L_halt + λ_type L_type",
                        "error_types": list(ERROR_TYPES),
                    }, indent=2))
                    print(f"[ckpt] {ckpt}", flush=True)
                if gstep >= total_steps:
                    break
        if gstep >= total_steps:
            break

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    torch.save({
        "halt_head": halt_head.state_dict(),
        "type_head": type_head.state_dict(),
        "error_types": list(ERROR_TYPES),
        "global_step": gstep,
        "base_model": args.model_path,
    }, Path(args.output_dir) / "aux_heads.bin")
    (Path(args.output_dir) / "config_task.json").write_text(json.dumps({
        "task": "finebio_protocol_prefix_stream_sft",
        "level": "protocol",
        "model_path": args.model_path,
        "global_step": gstep,
        "max_frames": args.max_frames,
        "loss": "L_lm + λ_halt L_halt + λ_type L_type",
        "error_types": list(ERROR_TYPES),
    }, indent=2))
    logf.close()
    print(f"[done] {args.output_dir} step={gstep}", flush=True)


if __name__ == "__main__":
    main()
