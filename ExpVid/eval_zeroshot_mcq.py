#!/usr/bin/env python3
"""Zero-shot / post-SSL MCQ eval of Qwen2.5-VL on ExpVid level-1 (image + options).

Uses a single mid-frame image per clip (not full video). Optionally prepends
asr_caption to the prompt (--use-caption) for image+caption conditioning.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
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


def build_prompt(rec: dict, use_caption: bool) -> str:
    opts = rec["options"]
    opt_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    parts = []
    if use_caption and rec.get("asr_caption"):
        parts.append(f"Narration caption: {rec['asr_caption']}")
    parts.append(rec["question"])
    parts.append("Options:")
    parts.append(opt_txt)
    parts.append("Answer with a single letter (A, B, C, or D) only.")
    return "\n".join(parts)


def parse_letter(text: str) -> str | None:
    t = (text or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    if t and t[0] in "ABCD":
        return t[0]
    return None


@torch.inference_mode()
def run_one(model, processor, image: Image.Image, prompt: str, device) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = processor(text=[text], images=[image], padding=True)
    inputs = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
        else:
            inputs[k] = torch.as_tensor(v).to(device)
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/ll5914/Labos/ExpVid")
    ap.add_argument("--qa-jsonl", default=None)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--checkpoint", default=None, help="Optional LoRA dir")
    ap.add_argument("--use-caption", action="store_true")
    ap.add_argument("--tasks", default="materials,operation,tools,quantity")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    qa_path = Path(args.qa_jsonl) if args.qa_jsonl else root / "processed" / "qa_level1.jsonl"
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    rows = []
    with qa_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") not in tasks:
                continue
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[eval] n={len(rows)} use_caption={args.use_caption} ckpt={args.checkpoint}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    configure_processor_pixels(processor, args.max_pixels, args.min_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    by_task = defaultdict(lambda: {"n": 0, "ok": 0})
    conf = Counter()
    results = []
    t0 = time.time()
    for i, rec in enumerate(rows, 1):
        img_path = root / rec["image_path"]
        image = Image.open(img_path).convert("RGB")
        prompt = build_prompt(rec, args.use_caption)
        reply = run_one(model, processor, image, prompt, device)
        pred = parse_letter(reply)
        gt = str(rec["answer"]).strip().upper()
        ok = pred == gt
        if ok:
            correct += 1
        by_task[rec["task"]]["n"] += 1
        by_task[rec["task"]]["ok"] += int(ok)
        conf[(gt, pred or "?")] += 1
        results.append({
            "id": rec["id"], "task": rec["task"], "gt": gt, "pred": pred,
            "ok": ok, "reply": reply,
        })
        if i % 50 == 0 or i == len(rows):
            print(
                f"[{i}/{len(rows)}] acc={correct/i:.3f} "
                f"{(time.time()-t0)/i:.2f}s/sample",
                flush=True,
            )

    summary = {
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "use_caption": args.use_caption,
        "checkpoint": args.checkpoint,
        "model_path": args.model_path,
        "by_task": {
            t: {"n": v["n"], "accuracy": v["ok"] / max(v["n"], 1)}
            for t, v in sorted(by_task.items())
        },
        "confusion_gt_pred": {f"{a}->{b}": c for (a, b), c in conf.items()},
        "seconds": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
