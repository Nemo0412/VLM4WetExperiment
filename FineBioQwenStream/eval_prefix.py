#!/usr/bin/env python3
"""Evaluate prefix SFT on val/test jsonl: decision + error_type accuracy."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from decord import VideoReader, cpu
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from protocol_prompt import parse_decision


def load_frames(video_path: str, indices: list[int]):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(i, 0), n - 1) for i in indices]
    return vr.get_batch(idxs).asnumpy()


@torch.inference_mode()
def run_one(model, processor, frames, user_text, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    scfg = json.loads((ckpt / "config_task.json").read_text()) if (ckpt / "config_task.json").exists() else {}
    base = scfg.get("model_path", args.model_path)

    processor = AutoProcessor.from_pretrained(
        str(ckpt) if (ckpt / "preprocessor_config.json").exists() else base,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt))
    model.eval()
    device = model.device

    rows = []
    with open(args.data_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    n = len(rows)
    dec_ok = et_ok = halt_n = 0
    conf = Counter()
    results = []
    t0 = time.time()
    print(f"[eval] n={n} ckpt={ckpt}", flush=True)

    for i, rec in enumerate(rows):
        path = str(Path(args.video_root) / rec["video"])
        frames = load_frames(path, rec["frame_indices"])
        user_text = rec["messages"][0]["content"]
        reply = run_one(model, processor, frames, user_text, device)
        pred_d, pred_et = parse_decision(reply)
        gt_d = "CONTINUE" if rec["label"] == "continue" else "HALT"
        gt_et = rec.get("error_type")
        d_ok = pred_d == gt_d
        e_ok = True
        if gt_d == "HALT":
            halt_n += 1
            e_ok = (pred_et == gt_et) if pred_et and gt_et else False
            if e_ok:
                et_ok += 1
        if d_ok:
            dec_ok += 1
        conf[(gt_d, pred_d)] += 1
        results.append({
            "id": rec["id"], "gt": gt_d, "gt_et": gt_et,
            "pred": pred_d, "pred_et": pred_et, "reply": reply,
            "dec_ok": d_ok, "et_ok": e_ok,
        })
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(
                f"[{i+1}/{n}] dec={dec_ok/(i+1):.3f} "
                f"et={et_ok/max(halt_n,1):.3f} {(time.time()-t0)/(i+1):.2f}s/sample",
                flush=True,
            )

    summary = {
        "n": n,
        "decision_accuracy": dec_ok / max(n, 1),
        "error_type_accuracy_on_halt": et_ok / max(halt_n, 1),
        "n_halt_gt": halt_n,
        "confusion_gt_pred": {f"{a}->{b}": c for (a, b), c in conf.items()},
        "seconds": time.time() - t0,
    }
    out = Path(args.out) if args.out else ckpt / "eval_prefix.json"
    out.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
