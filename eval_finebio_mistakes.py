#!/usr/bin/env python3
"""Compare zero-shot / baseline SFT / A+C SSL on FineBio mistake trials."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel

from infer_protocol_compliance import (
    build_prompt,
    make_grid,
    sample_video_frames,
)
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


CASES = [
    {
        "name": "P06_03",
        "proto": 3,
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P06_03_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/outputs/finebio_test/P06_03_02.mp4",
        "mistake_note": "missing sterile water wash",
    },
    {
        "name": "P17_02",
        "proto": 2,
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P17_02_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos/P17_02_02.mp4",
        "mistake_note": "extra PBS wash",
    },
    {
        "name": "P11_06",
        "proto": 6,
        "correct": "/scratch/ll5914/Labos/Llava/data/FineBio/videos_w640/P10_06_01.mp4",
        "mistake": "/scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos/P11_06_01.mp4",
        "mistake_note": "extra wash buffer",
    },
]


def protocol_text(proto_ref: dict, proto_id: int) -> str:
    info = proto_ref[str(proto_id)]
    lines = [f"Protocol {proto_id}: {info['name']}", "", "Steps:"]
    for i, s in enumerate(info["steps"], 1):
        lines.append(f"{i}. {s.replace('_', ' ')}")
    return "\n".join(lines)


def load_lora_llava(base_path: str, lora_dir: str):
    disable_torch_init()
    model_name = get_model_name_from_path(base_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base_path, None, model_name, device_map=None, torch_dtype=torch.bfloat16,
    )
    nl_path = Path(lora_dir) / "non_lora_trainables.bin"
    if nl_path.exists():
        raw = torch.load(nl_path, map_location="cpu")
        fixed = {}
        for k, v in raw.items():
            nk = k.replace("base_model.model.model.", "model.model.")
            fixed[nk] = v
        model.load_state_dict(fixed, strict=False)
    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()
    model.to("cuda")
    return tokenizer, model, image_processor


def parse_verdict(text: str) -> str:
    m = re.search(r"verdict:\s*(FOLLOWED|NOT FOLLOWED)", text, re.I)
    if m:
        return m.group(1).upper()
    if "NOT FOLLOWED" in text.upper():
        return "NOT FOLLOWED"
    if "FOLLOWED" in text.upper():
        return "FOLLOWED"
    return "UNKNOWN"


def run_one(model, tokenizer, image_processor, model_label, video, protocol, num_frames, max_new_tokens):
    frames = sample_video_frames(video, num_frames=num_frames)
    grid = make_grid(frames)
    model_name = get_model_name_from_path("liuhaotian/llava-v1.5-7b")
    prompt, _ = build_prompt(protocol, model_name, getattr(model.config, "mm_use_im_start_end", False))
    dtype = next(model.parameters()).dtype
    images_tensor = process_images([grid], image_processor, model.config).to(model.device, dtype=dtype)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=images_tensor, image_sizes=[grid.size],
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[-1].strip()
    return {"model": model_label, "video": video, "answer": answer, "verdict": parse_verdict(answer)}


def score_case(results):
    """1 if correct verdict on both videos, 0.5 if one right, 0 if both wrong."""
    by_type = {r["label"]: r["verdict"] for r in results}
    ok = 0
    if by_type.get("correct") == "FOLLOWED":
        ok += 1
    if by_type.get("mistake") == "NOT FOLLOWED":
        ok += 1
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--baseline-lora", default="/scratch/ll5914/Labos/Llava/outputs/finebio_lora_baseline")
    ap.add_argument("--ssl-lora", default="/scratch/ll5914/Labos/Llava/outputs/finebio_ssl_ac")
    ap.add_argument("--proto-ref", default="/scratch/ll5914/Labos/Llava/data/finebio_llava/protocol_reference.json")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/Llava/outputs/finebio_eval_compare")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--skip-zero-shot", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proto_ref = json.loads(Path(args.proto_ref).read_text())
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]

    models = []
    if not args.skip_zero_shot:
        models.append(("zero_shot", None))
    models.append(("baseline_sft", args.baseline_lora))
    models.append(("ssl_ac", args.ssl_lora))

    all_results = []
    summary = []

    for mlabel, lora_dir in models:
        print(f"\n===== loading {mlabel} =====", flush=True)
        if lora_dir is None:
            disable_torch_init()
            mn = get_model_name_from_path(args.base_model)
            tokenizer, model, image_processor, _ = load_pretrained_model(
                args.base_model, None, mn, device_map=None, torch_dtype=torch.bfloat16,
            )
            model.to("cuda")
        else:
            tokenizer, model, image_processor = load_lora_llava(args.base_model, lora_dir)

        for case in CASES:
            protocol = protocol_text(proto_ref, case["proto"])
            case_rows = []
            for label, vpath in [("correct", case["correct"]), ("mistake", case["mistake"])]:
                print(f"[{mlabel}] {case['name']} {label} ...", flush=True)
                row = run_one(model, tokenizer, image_processor, mlabel, vpath, protocol,
                              args.num_frames, 512)
                row.update({"case": case["name"], "label": label, "expected": "FOLLOWED" if label == "correct" else "NOT FOLLOWED",
                            "mistake_note": case["mistake_note"]})
                case_rows.append(row)
                all_results.append(row)
                out_file = out / f"{mlabel}_{case['name']}_{label}.txt"
                out_file.write_text(row["answer"] + "\n", encoding="utf-8")

            pts = score_case(case_rows)
            summary.append({"model": mlabel, "case": case["name"], "points": pts,
                            "correct_verdict": case_rows[0]["verdict"],
                            "mistake_verdict": case_rows[1]["verdict"]})
            print(f"  -> correct={case_rows[0]['verdict']} mistake={case_rows[1]['verdict']} score={pts}/2")

        del model
        torch.cuda.empty_cache()

    total = {}
    for s in summary:
        total[s["model"]] = total.get(s["model"], 0) + s["points"]
    report = {"summary": summary, "total_points_out_of_6": total, "cases": len(CASES)}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# FineBio Mistake Trial Comparison", "",
             "| Model | Case | Correct video | Mistake video | Score |",
             "|-------|------|---------------|---------------|-------|"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['case']} | {s['correct_verdict']} | {s['mistake_verdict']} | {s['points']}/2 |")
    lines += ["", "## Total (max 6)", ""]
    for m, sc in total.items():
        lines.append(f"- **{m}**: {sc}/6")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[-5:]), flush=True)
    print(f"[done] saved to {out}")


if __name__ == "__main__":
    main()
