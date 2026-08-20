#!/usr/bin/env python3
"""LoRA SFT for EgoProactive when2prompt (official starter_kit protocol)."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup, set_seed

from proactive_protocol import (
    build_messages,
    extract_cumulative_frames,
    video_file,
)


class ProactiveSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        video_folder: str,
        processor,
        *,
        frames_per_interval: int = 16,
        max_frames: int = 32,
        max_history_turns: int = 4,
        max_seq_length: int = 8192,
    ) -> None:
        with open(jsonl_path) as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        self.video_folder = video_folder
        self.processor = processor
        self.frames_per_interval = frames_per_interval
        self.max_frames = max_frames
        self.max_history_turns = max_history_turns
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def _to_mm_messages(self, frames: list[object], messages: list[dict[str, str]]) -> list[dict]:
        mm: list[dict] = []
        images_inserted = False
        for msg in messages:
            role, text = msg["role"], msg["content"]
            if role == "user" and not images_inserted and frames:
                content: list[dict] = [{"type": "image", "image": img} for img in frames]
                content.append({"type": "text", "text": text})
                mm.append({"role": role, "content": content})
                images_inserted = True
            else:
                mm.append({"role": role, "content": text})
        return mm

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        vp = video_file(self.video_folder, sample["video_path"])
        chunk_index = int(sample["chunk_index"])

        frames = extract_cumulative_frames(
            vp,
            sample["video_intervals"],
            chunk_index,
            frames_per_interval=self.frames_per_interval,
            max_frames=self.max_frames,
        )
        prompt_messages = build_messages(
            sample.get("query", ""),
            sample.get("dialog_at_chunk", []),
            max_history_turns=self.max_history_turns,
        )
        full_messages = prompt_messages + [
            {"role": "assistant", "content": sample["target"]}
        ]

        for keep in [len(frames), max(2, len(frames) // 2), 2, 0]:
            use_frames = frames[:keep] if keep else []
            try:
                return self._encode(sample, use_frames, prompt_messages, full_messages)
            except Exception:
                continue
        return self._encode(sample, [], prompt_messages, full_messages)

    def _encode(
        self,
        sample: dict,
        frames: list[object],
        prompt_messages: list[dict[str, str]],
        full_messages: list[dict[str, str]],
    ) -> dict:
        prompt_mm = self._to_mm_messages(frames, prompt_messages)
        full_mm = self._to_mm_messages(frames, full_messages)

        full_text = self.processor.apply_chat_template(
            full_mm, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_mm, tokenize=False, add_generation_prompt=True
        )

        if frames:
            full_inputs = self.processor(
                text=[full_text], images=frames, padding=False, return_tensors="pt"
            )
            prompt_inputs = self.processor(
                text=[prompt_text], images=frames, padding=False, return_tensors="pt"
            )
        else:
            full_inputs = self.processor(text=[full_text], padding=False, return_tensors="pt")
            prompt_inputs = self.processor(text=[prompt_text], padding=False, return_tensors="pt")

        input_ids = full_inputs["input_ids"].squeeze(0)
        if input_ids.shape[0] > self.max_seq_length:
            raise RuntimeError("sequence too long")

        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[: min(prompt_len, labels.shape[0])] = -100

        item = {k: v.squeeze(0) for k, v in full_inputs.items() if k != "labels"}
        item["labels"] = labels
        item["decision"] = sample.get("decision", "")
        return item


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in batch[0]:
            continue
        padded = []
        for x in batch:
            t = x[key]
            pad_len = max_len - t.shape[0]
            if key == "labels":
                pad = torch.full((pad_len,), -100, dtype=t.dtype)
            elif key == "input_ids":
                pad = torch.full((pad_len,), pad_token_id, dtype=t.dtype)
            else:
                pad = torch.zeros(pad_len, dtype=t.dtype)
            padded.append(torch.cat([t, pad]))
        out[key] = torch.stack(padded)
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.cat([x["pixel_values"] for x in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in batch], dim=0)
    out["decisions"] = [x.get("decision", "") for x in batch]
    return out


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch.pop("decisions", None)
            outputs = model(**batch)
            losses.append(float(outputs.loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(cfg: dict) -> None:
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["training"]
    out_dir = tcfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    set_seed(int(tcfg.get("seed", 42)))

    with open(os.path.join(out_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    processor = AutoProcessor.from_pretrained(mcfg["name"])
    if hasattr(processor, "image_processor") and mcfg.get("image_max_pixels"):
        processor.image_processor.max_pixels = int(mcfg["image_max_pixels"])
        processor.image_processor.min_pixels = int(mcfg.get("image_min_pixels", 50176))

    model = AutoModelForVision2Seq.from_pretrained(
        mcfg["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(mcfg["lora_rank"]),
        lora_alpha=int(mcfg["lora_alpha"]),
        lora_dropout=float(mcfg.get("lora_dropout", 0.05)),
        target_modules=mcfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    common = dict(
        video_folder=dcfg["video_folder"],
        processor=processor,
        frames_per_interval=int(dcfg.get("frames_per_interval", 16)),
        max_frames=int(dcfg.get("max_frames", 32)),
        max_history_turns=int(dcfg.get("max_history_turns", 4)),
        max_seq_length=int(tcfg.get("max_seq_length", 8192)),
    )
    train_ds = ProactiveSFTDataset(dcfg["train_jsonl"], **common)
    val_ds = ProactiveSFTDataset(dcfg["val_jsonl"], **common)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=int(dcfg.get("num_workers", 2)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=int(dcfg.get("num_workers", 2)),
    )

    device = model.device
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.1)),
    )
    accum = int(tcfg.get("gradient_accumulation_steps", 16))
    total_steps = (len(train_loader) // accum + 1) * int(tcfg.get("num_epochs", 2))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(tcfg.get("warmup_steps", 100)),
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    best_dir = os.path.join(out_dir, "best")
    step = 0
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    for epoch in range(int(tcfg.get("num_epochs", 2))):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        optim.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch.pop("decisions", None)
            outputs = model(**batch)
            loss = outputs.loss / accum
            loss.backward()
            if (batch_idx + 1) % accum == 0:
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "train_loss": float(outputs.loss), "lr": sched.get_last_lr()[0]}
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                pbar.set_postfix(loss=f"{outputs.loss:.4f}")

                if step % int(tcfg.get("eval_steps", 250)) == 0:
                    val_loss = evaluate(model, val_loader, device)
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                    print(f"[train] step={step} val_loss={val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        model.save_pretrained(best_dir)
                        processor.save_pretrained(best_dir)
                        print(f"[train] saved best -> {best_dir}")

    final_dir = os.path.join(out_dir, "last")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[train] done best_val={best_val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
