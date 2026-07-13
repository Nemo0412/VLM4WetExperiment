#!/usr/bin/env python3
"""FineBioStreaming multi-GPU trainer (torchrun / DDP).

VLM backbone: LLaVA-NeXT-Video-7B-DPO
Loss: L = L_lm + λ_halt L_halt + λ_step L_step

Features:
  - 4-GPU DDP via torchrun
  - periodic checkpoint saves (resume-safe)
  - DataLoader workers to keep GPUs fed (higher utilization)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig

import tokenizers
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

IS_TOKENIZER_GT_014 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def preprocess_multimodal(sources: Sequence, use_im_start_end: bool = False) -> Sequence:
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = (
                    DEFAULT_IMAGE_TOKEN + "\n"
                    + sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
            tok = DEFAULT_IMAGE_TOKEN
            if use_im_start_end:
                tok = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, tok)
    return sources


def preprocess_v1(sources, tokenizer, has_image: bool = True):
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

    input_ids = torch.stack(
        [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations], dim=0)
    targets = input_ids.clone()
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
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GT_014:
                round_len -= 1
                instruction_len -= 1
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def load_video_tensor(video_path: str, indices: list[int], image_processor):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idxs = [min(max(i, 0), total - 1) for i in indices]
    frames = vr.get_batch(idxs).asnumpy()
    tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    h, w = int(frames.shape[1]), int(frames.shape[2])
    return tensor, (w, h)


class StreamingVLMDataset(Dataset):
    def __init__(self, json_path, video_folder, tokenizer, image_processor):
        self.data = json.load(open(json_path))
        self.video_folder = video_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        path = os.path.join(self.video_folder, rec["video"])
        pixel, size = load_video_tensor(path, rec["frame_indices"], self.image_processor)
        sources = preprocess_multimodal(copy.deepcopy([rec["conversations"]]))
        tok = preprocess_v1(sources, self.tokenizer, has_image=True)
        return {
            "input_ids": tok["input_ids"][0],
            "labels": tok["labels"][0],
            "image": [(pixel, size, "video")],
            "halt_label": int(rec["halt_label"]),
            "step_id": int(rec.get("expected_step_id", -100)),
        }


@dataclass
class Collator:
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
            "halt_label": torch.tensor([b["halt_label"] for b in batch], dtype=torch.long),
            "step_id": torch.tensor([b["step_id"] for b in batch], dtype=torch.long),
        }


def find_lora_targets(model):
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(k in name for k in ["vision_tower", "mm_projector", "lm_head", "halt_head", "step_head"]):
            continue
        names.add(name.split(".")[-1])
    return sorted(names)


def build_overwrite_config(model_path: str, num_frames: int, pool_stride: int = 2) -> dict:
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


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(out_dir: Path, model, tokenizer, halt_head, step_head, args, gstep, optim=None):
    ckpt_dir = out_dir / f"checkpoint-{gstep}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    m = unwrap(model)
    m.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    non_lora = {n: p.detach().cpu() for n, p in m.named_parameters()
                if p.requires_grad and "lora_" not in n}
    torch.save(non_lora, ckpt_dir / "non_lora_trainables.bin")
    torch.save({
        "halt_head": halt_head.state_dict(),
        "step_head": step_head.state_dict(),
        "n_halt": args.n_halt,
        "n_step": args.n_step,
        "num_frames": args.num_frames,
        "pool_stride": args.pool_stride,
        "base_model": args.model_path,
        "global_step": gstep,
    }, ckpt_dir / "aux_heads.bin")
    if optim is not None:
        torch.save(optim.state_dict(), ckpt_dir / "optimizer.pt")
    (ckpt_dir / "config_streaming.json").write_text(json.dumps({
        "architecture": "vlm_llava_next_video",
        "model_path": args.model_path,
        "num_frames": args.num_frames,
        "n_halt": args.n_halt,
        "n_step": args.n_step,
        "global_step": gstep,
        "halt_labels": {
            "continue": 0, "missing_step": 1, "redundant_step": 2,
            "wrong_order": 3, "within_step_error": 4,
        },
    }, indent=2))
    # Also refresh "latest" pointer
    latest = out_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)
    print(f"[ckpt] saved {ckpt_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--video-folder", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conv-version", default="vicuna_v1")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--proj-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lambda-halt", type=float, default=1.0)
    ap.add_argument("--lambda-step", type=float, default=0.3)
    ap.add_argument("--n-halt", type=int, default=5)
    ap.add_argument("--n-step", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, local_rank, world_size = setup_dist()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # Keep GPU context warm (helps avoid idle-killer false positives at startup)
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)

    if args.n_step <= 0:
        vocab = json.load(open(Path(args.video_folder) / "step_vocab.json"))
        args.n_step = len(vocab)

    overwrite = build_overwrite_config(args.model_path, args.num_frames, args.pool_stride)
    model_name = get_model_name_from_path(args.model_path)
    # Important for DDP: load onto this rank's GPU only (no device_map=auto)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name,
        device_map={"": local_rank},
        torch_dtype="bfloat16",
        overwrite_config=overwrite,
        attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_version]

    from peft import LoraConfig, get_peft_model
    targets = find_lora_targets(model)
    if is_main():
        print(f"[lora] targets={targets} world_size={world_size}", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    ))
    base = model.get_base_model()
    for n, p in base.named_parameters():
        if "mm_projector" in n:
            p.requires_grad_(True)
    for p in base.get_vision_tower().parameters():
        p.requires_grad_(False)

    hidden = model.config.hidden_size
    halt_head = nn.Linear(hidden, args.n_halt, device=device, dtype=torch.bfloat16)
    step_head = nn.Linear(hidden, args.n_step, device=device, dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        halt_head = DDP(halt_head, device_ids=[local_rank], output_device=local_rank)
        step_head = DDP(step_head, device_ids=[local_rank], output_device=local_rank)

    ds = StreamingVLMDataset(args.train_json, args.video_folder, tokenizer, image_processor)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=Collator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )

    if is_main():
        s0 = ds[0]
        n_sup = int((s0["labels"] != IGNORE_INDEX).sum())
        print(f"[sanity] supervised_tokens={n_sup}", flush=True)
        if n_sup == 0:
            raise RuntimeError("Label mask broken")

    raw_model = unwrap(model)
    lora_params, proj_params = [], []
    for n, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (proj_params if "mm_projector" in n else lora_params).append(p)
    head_params = list(unwrap(halt_head).parameters()) + list(unwrap(step_head).parameters())
    optim = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": proj_params, "lr": args.proj_lr},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=0.0)

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: (s / warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    if is_main():
        print(f"[train] VLM={args.model_path} samples={len(ds)} "
              f"gpus={world_size} frames={args.num_frames} total_steps={total_steps} "
              f"save_every={args.save_every}", flush=True)
        logf = open(Path(args.output_dir) / "train_log.jsonl", "a")
    else:
        logf = None

    gstep = micro = 0
    last_busy = time.time()
    optim.zero_grad(set_to_none=True)

    for epoch in range(math.ceil(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            input_ids = batch["input_ids"][:, :args.max_len].to(device, non_blocking=True)
            labels = batch["labels"][:, :args.max_len].to(device, non_blocking=True)
            attn = batch["attention_mask"][:, :args.max_len].to(device, non_blocking=True)
            images = [v.to(device=device, dtype=torch.bfloat16, non_blocking=True) for v in batch["images"]]
            halt_y = batch["halt_label"].to(device, non_blocking=True)
            step_y = batch["step_id"].to(device, non_blocking=True)

            out = model(
                input_ids=input_ids, attention_mask=attn, labels=labels,
                images=images, modalities=batch["modalities"],
                image_sizes=batch["image_sizes"],
                output_hidden_states=True, return_dict=True,
            )
            lm_loss = out.loss.float() if out.loss is not None else torch.tensor(float("nan"), device=device)
            if not torch.isfinite(lm_loss):
                if is_main():
                    print("[warn] non-finite lm_loss, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            z = out.hidden_states[-1][:, -1, :].to(torch.bfloat16)
            # Upweight non-CONTINUE classes so halt head does not collapse
            halt_w = torch.tensor([1.0, 2.5, 2.5, 2.5, 2.5], device=device, dtype=torch.float32)
            if halt_w.numel() != args.n_halt:
                halt_w = torch.ones(args.n_halt, device=device, dtype=torch.float32)
                if args.n_halt > 1:
                    halt_w[1:] = 2.5
            l_halt = F.cross_entropy(unwrap(halt_head)(z).float(), halt_y, weight=halt_w)
            l_step = F.cross_entropy(unwrap(step_head)(z).float(), step_y, ignore_index=-100)
            total = lm_loss + args.lambda_halt * l_halt + args.lambda_step * l_step
            if not torch.isfinite(total):
                if is_main():
                    print("[warn] non-finite total, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            (total / args.grad_accum).backward()
            micro += 1
            last_busy = time.time()

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + proj_params + head_params, 0.5)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1

                # Lightweight GPU heartbeat every step (prints every log_every)
                if is_main() and gstep % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    rec = {
                        "step": gstep, "loss": float(total), "lm": float(lm_loss),
                        "halt": float(l_halt), "step_ce": float(l_step),
                        "gpu_mem_gb": round(mem, 2), "gpus": world_size,
                    }
                    print(
                        f"[step {gstep}/{total_steps}] loss={rec['loss']:.4f} "
                        f"lm={rec['lm']:.4f} halt={rec['halt']:.4f} step={rec['step_ce']:.4f} "
                        f"mem={rec['gpu_mem_gb']:.1f}GB",
                        flush=True,
                    )
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()

                if is_main() and args.save_every > 0 and gstep % args.save_every == 0:
                    save_checkpoint(
                        Path(args.output_dir), model, tokenizer,
                        unwrap(halt_head), unwrap(step_head), args, gstep, optim,
                    )
                    # Touch a heartbeat file so idle monitors see activity
                    (Path(args.output_dir) / "heartbeat").write_text(
                        f"step={gstep} t={time.time()}\n")

            if gstep >= total_steps:
                break
        if gstep >= total_steps:
            break

    if is_main():
        save_checkpoint(
            Path(args.output_dir), model, tokenizer,
            unwrap(halt_head), unwrap(step_head), args, gstep, optim,
        )
        # Also dump final flat copy for infer_stream.py
        final = Path(args.output_dir)
        m = unwrap(model)
        m.save_pretrained(str(final))
        tokenizer.save_pretrained(str(final))
        torch.save({
            "halt_head": unwrap(halt_head).state_dict(),
            "step_head": unwrap(step_head).state_dict(),
            "n_halt": args.n_halt,
            "n_step": args.n_step,
            "num_frames": args.num_frames,
            "pool_stride": args.pool_stride,
            "base_model": args.model_path,
            "global_step": gstep,
        }, final / "aux_heads.bin")
        (final / "config_streaming.json").write_text(json.dumps({
            "architecture": "vlm_llava_next_video",
            "model_path": args.model_path,
            "num_frames": args.num_frames,
            "n_halt": args.n_halt,
            "n_step": args.n_step,
            "global_step": gstep,
        }, indent=2))
        if logf:
            logf.close()
        print(f"[done] saved -> {final} (step={gstep})", flush=True)

    cleanup_dist()


if __name__ == "__main__":
    main()
