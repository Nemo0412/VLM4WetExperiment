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
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from frame_cache import load_frames_cached
from protocol_prompt import ERROR_TYPES

# Set by USR1/TERM trap so the train loop can checkpoint and exit cleanly.
_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] got {signum}; will stop after next optimizer step", flush=True)


def load_concat_segments(
    video_root: str,
    segments: list[dict],
    cache_dir: str | None = None,
) -> tuple[np.ndarray, int, int]:
    """Concatenate frames; returns (frames, n_hits, n_misses)."""
    parts = []
    hits = misses = 0
    for seg in segments:
        path = os.path.join(video_root, seg["video"])
        frames, hit = load_frames_cached(
            path,
            list(seg["frame_indices"]),
            cache_dir=cache_dir,
            video_rel=seg["video"],
        )
        parts.append(frames)
        if hit:
            hits += 1
        else:
            misses += 1
    return np.concatenate(parts, axis=0), hits, misses


class ProtoPrefixDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        video_root: str,
        max_frames: int = 128,
        frame_cache_dir: str | None = None,
    ):
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.video_root = video_root
        self.max_frames = max_frames
        self.frame_cache_dir = frame_cache_dir
        n_cache = 0
        if frame_cache_dir and Path(frame_cache_dir).exists():
            n_cache = sum(1 for _ in Path(frame_cache_dir).glob("*.npy"))
        print(
            f"[data] {jsonl_path}: {len(self.rows)} samples "
            f"frame_cache={frame_cache_dir} npy={n_cache}",
            flush=True,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rec = self.rows[i]
        t0 = time.perf_counter()
        frames, hits, misses = load_concat_segments(
            self.video_root,
            rec["segments"],
            cache_dir=self.frame_cache_dir,
        )
        decode_wait = time.perf_counter() - t0
        if len(frames) > self.max_frames:
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
            "batch_wait": decode_wait,
            "cache_hits": hits,
            "cache_misses": misses,
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
    # Do not pass return_tensors/fps — some Qwen processor builds reject them.
    raw = processor(text=[text], videos=[frames], padding=True)
    inputs = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            inputs[k] = v
        elif isinstance(v, np.ndarray):
            inputs[k] = torch.from_numpy(v)
        else:
            inputs[k] = torch.as_tensor(v)

    user_only = [{"role": "user", "content": messages[0]["content"]}]
    prompt_text = processor.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    prompt_out = processor(text=[prompt_text], videos=[frames])
    prompt_ids = prompt_out["input_ids"]
    if not torch.is_tensor(prompt_ids):
        prompt_ids = torch.as_tensor(prompt_ids)
    labels = inputs["input_ids"].clone()
    labels[:, : prompt_ids.shape[1]] = -100
    inputs["labels"] = labels
    inputs["halt_label"] = torch.tensor([b0["halt_label"]], dtype=torch.long)
    inputs["type_id"] = torch.tensor([b0["type_id"]], dtype=torch.long)
    inputs["loss_weight"] = torch.tensor([b0["loss_weight"]], dtype=torch.float32)
    inputs["halt_horizon"] = b0.get("halt_horizon")
    inputs["batch_wait"] = float(b0.get("batch_wait", 0.0))
    inputs["cache_hits"] = int(b0.get("cache_hits", 0))
    inputs["cache_misses"] = int(b0.get("cache_misses", 0))
    return inputs


def find_latest_ckpt(output_dir: Path) -> Path | None:
    latest = output_dir / "latest"
    if (latest / "aux_heads.bin").exists() or (latest / "adapter_config.json").exists():
        return latest
    cands = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return cands[-1] if cands else None


def save_checkpoint(
    output_dir: Path,
    model,
    processor,
    halt_head,
    type_head,
    optim,
    sched,
    gstep: int,
    base_model: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    tag: str | None = None,
):
    name = tag or f"checkpoint-{gstep}"
    ckpt = output_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    torch.save(
        {
            "halt_head": halt_head.state_dict(),
            "type_head": type_head.state_dict(),
            "error_types": list(ERROR_TYPES),
            "global_step": gstep,
            "base_model": base_model,
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
        },
        ckpt / "aux_heads.bin",
    )
    (ckpt / "config_task.json").write_text(
        json.dumps(
            {
                "task": "finebio_protocol_prefix_stream_sft",
                "level": "protocol",
                "model_path": base_model,
                "global_step": gstep,
                "max_frames": max_frames,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "loss": "L_lm + λ_halt L_halt + λ_type L_type",
                "error_types": list(ERROR_TYPES),
            },
            indent=2,
        )
    )
    # Keep a rolling "latest" pointer for auto-resume (copy via save again if tag!=latest)
    if name != "latest":
        latest = output_dir / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(latest))
        processor.save_pretrained(str(latest))
        torch.save(
            {
                "halt_head": halt_head.state_dict(),
                "type_head": type_head.state_dict(),
                "error_types": list(ERROR_TYPES),
                "global_step": gstep,
                "base_model": base_model,
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
            },
            latest / "aux_heads.bin",
        )
        (latest / "config_task.json").write_text((ckpt / "config_task.json").read_text())
    print(f"[ckpt] {ckpt}", flush=True)
    return ckpt


def find_lora_targets(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
                names.add(leaf)
    return sorted(names) or ["q_proj", "v_proj"]


def configure_processor_pixels(processor, max_pixels: int, min_pixels: int):
    """Cap Qwen2.5-VL vision tokens — default max_pixels≈12M OOMs with video."""
    for proc in (getattr(processor, "image_processor", None),
                 getattr(processor, "video_processor", None)):
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


def attach_last_hidden_hook(model):
    """Capture only the last decoder layer hidden states (avoid storing all layers)."""
    bucket: dict = {}

    def _hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        bucket["h"] = h

    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = None
    # Try common Qwen2.5-VL paths, then fall back to any ModuleList named "layers"
    candidates = []
    core = getattr(root, "model", root)
    for obj in (
        getattr(getattr(core, "language_model", None), "layers", None),
        getattr(getattr(core, "model", None), "layers", None),
        getattr(core, "layers", None),
    ):
        if obj is not None:
            candidates.append(obj)
    if not candidates:
        for name, mod in root.named_modules():
            if name.endswith("language_model.layers") or name.endswith(".layers"):
                if hasattr(mod, "__len__") and len(mod) > 0:
                    candidates.append(mod)
                    break
    if not candidates:
        raise RuntimeError("Could not locate decoder layers for last-hidden hook")
    layers = candidates[0]
    print(f"[hook] last decoder layer among {len(layers)} layers", flush=True)
    handle = layers[-1].register_forward_hook(_hook)
    return bucket, handle


def pooled_last_hidden(h: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden state → [B, H] for aux heads."""
    return h.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frame-cache-dir", default=None,
                    help="Prefetched uint8 clips; skips VideoReader open on hit")
    ap.add_argument("--max-frames", type=int, default=8,
                    help="Cap concatenated frames after downsample (default 8)")
    ap.add_argument("--max-pixels", type=int, default=128*28*28,
                    help="Per-frame/video pixel budget for Qwen VL processor")
    ap.add_argument("--min-pixels", type=int, default=4*28*28)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lambda-halt", type=float, default=0.5,
                    help="Weight for explicit CONTINUE/HALT CE")
    ap.add_argument("--lambda-type", type=float, default=1.0,
                    help="Weight for explicit mistake-type CE (HALT only)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", default="auto",
                    help="auto|none|/path/to/ckpt  (auto = latest or newest checkpoint-*)")
    args = ap.parse_args()

    assert args.batch_size == 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Graceful stop for SLURM --signal=B:USR1@90 / util-kill TERM
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print(f"[load] {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    configure_processor_pixels(processor, args.max_pixels, args.min_pixels)
    print(
        f"[vision] max_frames={args.max_frames} max_pixels={args.max_pixels} "
        f"min_pixels={args.min_pixels} frame_cache={args.frame_cache_dir}",
        flush=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    resume_dir = None
    if args.resume == "auto":
        resume_dir = find_latest_ckpt(out_dir)
    elif args.resume and args.resume not in {"none", "None", ""}:
        resume_dir = Path(args.resume)

    if resume_dir is not None and (resume_dir / "adapter_config.json").exists():
        print(f"[resume] LoRA from {resume_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
    else:
        targets = find_lora_targets(model)
        print(f"[lora] targets={targets} r={args.lora_r}", flush=True)
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
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    last_h, last_h_handle = attach_last_hidden_hook(model)

    ds = ProtoPrefixDataset(
        args.train_jsonl,
        args.video_root,
        args.max_frames,
        frame_cache_dir=args.frame_cache_dir,
    )

    def _collate(batch):
        return collate_one(batch, processor)

    dl_kwargs = dict(
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=False,  # avoid host RAM spikes with video batches
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor
    dl = DataLoader(ds, **dl_kwargs)

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

    gstep = 0
    if resume_dir is not None and (resume_dir / "aux_heads.bin").exists():
        state = torch.load(resume_dir / "aux_heads.bin", map_location="cpu", weights_only=False)
        halt_head.load_state_dict(state["halt_head"])
        type_head.load_state_dict(state["type_head"])
        gstep = int(state.get("global_step", 0))
        if "optim" in state:
            try:
                optim.load_state_dict(state["optim"])
            except Exception as e:
                print(f"[resume] optim skip: {e}", flush=True)
        if "sched" in state:
            try:
                sched.load_state_dict(state["sched"])
            except Exception as e:
                print(f"[resume] sched skip: {e}", flush=True)
        print(f"[resume] aux_heads step={gstep} from {resume_dir}", flush=True)

    print(
        f"[train] protocol-level SFT  L=L_lm+{args.lambda_halt}*L_halt+{args.lambda_type}*L_type  "
        f"steps={total_steps} start={gstep} error_types={list(ERROR_TYPES)}",
        flush=True,
    )
    logf = open(out_dir / "train_log.jsonl", "a")
    micro = 0
    wait_sum = step_sum = 0.0
    hit_sum = miss_sum = wait_n = 0
    optim.zero_grad(set_to_none=True)
    finished = False

    for epoch in range(math.ceil(args.epochs)):
        if gstep >= total_steps or _STOP_REQUESTED:
            break
        for batch in dl:
            if gstep >= total_steps or _STOP_REQUESTED:
                break
            batch_wait = float(batch.pop("batch_wait", 0.0))
            hit_sum += int(batch.pop("cache_hits", 0))
            miss_sum += int(batch.pop("cache_misses", 0))
            halt_y = batch.pop("halt_label").to(device)
            type_y = batch.pop("type_id").to(device)
            w = batch.pop("loss_weight").to(device).float().mean()
            batch.pop("halt_horizon", None)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            t_step = time.perf_counter()
            last_h.pop("h", None)
            out = model(**batch, output_hidden_states=False)
            lm_loss = out.loss
            if lm_loss is None or not torch.isfinite(lm_loss):
                print("[warn] bad lm_loss, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            if "h" not in last_h:
                print("[warn] missing last hidden, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue
            z = pooled_last_hidden(last_h["h"]).to(torch.bfloat16)
            l_halt = F.cross_entropy(halt_head(z).float(), halt_y)
            mask = type_y >= 0
            if mask.any():
                l_type = F.cross_entropy(type_head(z).float()[mask], type_y[mask])
            else:
                l_type = torch.zeros((), device=device)

            base = lm_loss + args.lambda_halt * l_halt + args.lambda_type * l_type
            total = w * base
            if not torch.isfinite(total):
                print("[warn] bad total, skip", flush=True)
                optim.zero_grad(set_to_none=True)
                micro += 1
                continue

            (total / args.grad_accum).backward()
            last_h.pop("h", None)
            micro += 1
            wait_sum += batch_wait
            step_sum += time.perf_counter() - t_step
            wait_n += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 0.5)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    avg_wait = wait_sum / max(wait_n, 1)
                    avg_step = step_sum / max(wait_n, 1)
                    hit_rate = hit_sum / max(hit_sum + miss_sum, 1)
                    wait_sum = step_sum = 0.0
                    hit_sum = miss_sum = wait_n = 0
                    msg = {
                        "step": gstep,
                        "loss": float(total),
                        "lm": float(lm_loss),
                        "halt": float(l_halt),
                        "type": float(l_type),
                        "w": float(w),
                        "mem_gb": round(mem, 2),
                        "batch_wait_s": round(avg_wait, 3),
                        "step_s": round(avg_step, 3),
                        "hit_rate": round(hit_rate, 3),
                    }
                    print(
                        f"[step {gstep}/{total_steps}] loss={total:.4f} "
                        f"lm={lm_loss:.4f} halt={l_halt:.4f} type={l_type:.4f} "
                        f"w={float(w):.2f} mem={mem:.1f}GB "
                        f"wait={avg_wait:.3f}s step={avg_step:.3f}s "
                        f"hit_rate={hit_rate:.2f}",
                        flush=True,
                    )
                    logf.write(json.dumps(msg) + "\n")
                    logf.flush()
                if gstep % args.save_every == 0 or gstep >= total_steps or _STOP_REQUESTED:
                    save_checkpoint(
                        out_dir, model, processor, halt_head, type_head, optim, sched,
                        gstep, args.model_path, args.max_frames, args.max_pixels, args.min_pixels,
                    )
                if gstep >= total_steps:
                    finished = True
                    break
                if _STOP_REQUESTED:
                    print(f"[signal] stopping at step={gstep}", flush=True)
                    break
        if finished or _STOP_REQUESTED or gstep >= total_steps:
            break

    last_h_handle.remove()
    if finished:
        model.save_pretrained(str(out_dir))
        processor.save_pretrained(str(out_dir))
        torch.save({
            "halt_head": halt_head.state_dict(),
            "type_head": type_head.state_dict(),
            "error_types": list(ERROR_TYPES),
            "global_step": gstep,
            "base_model": args.model_path,
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
        }, out_dir / "aux_heads.bin")
        (out_dir / "config_task.json").write_text(json.dumps({
            "task": "finebio_protocol_prefix_stream_sft",
            "level": "protocol",
            "model_path": args.model_path,
            "global_step": gstep,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
            "loss": "L_lm + λ_halt L_halt + λ_type L_type",
            "error_types": list(ERROR_TYPES),
        }, indent=2))
        (out_dir / "TRAINING_DONE").write_text(f"step={gstep}\n")
        print(f"[done] {out_dir} step={gstep}", flush=True)
    else:
        # Ensure latest exists even if we stopped mid-interval
        if not (out_dir / "latest" / "aux_heads.bin").exists() or _STOP_REQUESTED:
            save_checkpoint(
                out_dir, model, processor, halt_head, type_head, optim, sched,
                gstep, args.model_path, args.max_frames, args.max_pixels, args.min_pixels,
                tag="latest",
            )
        print(f"[paused] {out_dir} step={gstep} (resume next segment)", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
