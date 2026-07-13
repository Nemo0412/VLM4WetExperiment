#!/usr/bin/env python3
"""Correct vs mistake streaming eval with LM-primary decisions + streaming metrics.

Decision modes
  lm     : halt only if generated text says HALT (default; fixes head false positives)
  head   : halt only if P(halt) >= threshold
  agree  : halt only if LM and head both say halt
  either : halt if LM or head says halt (old behavior)

Always records a full per-chunk trace. Streaming score uses first_halt under --decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
from peft import PeftModel

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from infer_stream import (
    build_overwrite_config,
    build_prompt,
    load_chunk,
    parse_text_halt,
)

FB = "/scratch/ll5914/Labos/Llava/data/FineBio"
VID = f"{FB}/videos_w640"
MIS = f"{FB}/mistake_videos"
ANN = f"{FB}/action_annotations"

# Unified fixed-window chunking for all splits (fair streaming comparison).
# gt_halt: approximate observability window from paired correct annotations.
CASES = [
    {
        "name": "P06_03",
        "proto": 3,
        "error_type": "missing_step",
        "note": "missing sterile water wash",
        "correct": f"{VID}/P06_03_01.mp4",
        "mistake": f"{MIS}/P06_03_02.mp4",
        # sterile water on correct ~218–228s; error observable around then onward
        "gt_halt": {"t0": 200.0, "t1": 280.0},
    },
    {
        "name": "P17_02",
        "proto": 2,
        "error_type": "redundant_step",
        "note": "extra PBS wash",
        "correct": f"{VID}/P17_02_01.mp4",
        "mistake": f"{MIS}/P17_02_02.mp4",
        # second PBS on correct ~73–99s; extra wash becomes visible mid protocol
        "gt_halt": {"t0": 55.0, "t1": 140.0},
    },
    {
        "name": "P11_06",
        "proto": 6,
        "error_type": "redundant_step",
        "note": "extra wash buffer (correct ref: P10_06_01)",
        "correct": f"{VID}/P10_06_01.mp4",
        "mistake": f"{MIS}/P11_06_01.mp4",
        # second wash buffer on P10 ~180–203s; scale down for shorter mistake (~188s)
        "gt_halt": {"t0": 100.0, "t1": 188.0},
    },
]


def build_fixed_chunks(video: str, window_sec: float):
    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps() or 30.0)
    duration = len(vr) / fps
    chunks = []
    t = 0.0
    while t < duration - 1e-6:
        chunks.append({"t0": t, "t1": min(t + window_sec, duration)})
        t += window_sec
    return chunks, fps, duration


def decide_halt(text_halt: bool, p_halt: float | None, threshold: float, mode: str) -> bool:
    head_halt = p_halt is not None and p_halt >= threshold
    if mode == "lm":
        return text_halt
    if mode == "head":
        return head_halt
    if mode == "agree":
        return text_halt and head_halt
    if mode == "either":
        return text_halt or head_halt
    raise ValueError(f"unknown decision mode: {mode}")


def run_stream(
    model, tokenizer, image_processor, halt_head, video, protocol_id,
    num_frames, window_sec, halt_threshold, conv_mode, decision,
):
    chunks, fps, duration = build_fixed_chunks(video, window_sec)
    history = []
    first_halt = None
    progress_steps: list[str] = []
    device = model.device

    with torch.inference_mode():
        for ci, ch in enumerate(chunks):
            video_t = load_chunk(video, ch["t0"], ch["t1"], fps, num_frames, image_processor)
            video_t = video_t.to(device, dtype=torch.bfloat16)
            cur_desc = f"time {ch['t0']:.1f}-{ch['t1']:.1f}s"
            prompt, conv = build_prompt(
                protocol_id, conv_mode,
                history_steps=progress_steps,
                current_desc=cur_desc,
            )
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            attn = input_ids.ne(tokenizer.pad_token_id or 0).long()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            p_halt = None
            pred_cls = None
            if halt_head is not None:
                out = model(
                    input_ids=input_ids, attention_mask=attn,
                    images=[video_t], modalities=["video"],
                    output_hidden_states=True, return_dict=True,
                )
                z = out.hidden_states[-1][:, -1, :].to(torch.bfloat16)
                probs = F.softmax(halt_head(z).float(), dim=-1)[0]
                p_halt = float(1.0 - probs[0].item())
                pred_cls = int(probs.argmax().item())

            output_ids = model.generate(
                inputs=input_ids, images=[video_t], attention_mask=attn,
                modalities="video", do_sample=False, temperature=1e-5,
                max_new_tokens=128, use_cache=True, stopping_criteria=[stopping],
            )
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            text_halt, _ = parse_text_halt(text)
            halted = decide_halt(text_halt, p_halt, halt_threshold, decision)

            rec = {
                "chunk": ci,
                "t0": ch["t0"],
                "t1": ch["t1"],
                "text": text,
                "text_halt": text_halt,
                "p_halt": p_halt,
                "pred_cls": pred_cls,
                "halted": halted,
                "history_steps": list(progress_steps),
            }
            history.append(rec)
            progress_steps.append(f"{ch['t0']:.0f}-{ch['t1']:.0f}s")
            if halted and first_halt is None:
                first_halt = {
                    "chunk": ci,
                    "t0": ch["t0"],
                    "t1": ch["t1"],
                    "reason": text,
                    "p_halt": p_halt,
                    "text_halt": text_halt,
                }

    verdict = "HALT" if first_halt is not None else "CONTINUE"
    return {
        "verdict": verdict,
        "decision": decision,
        "halt_threshold": halt_threshold,
        "first_halt": first_halt,
        "halt_chunk": first_halt["chunk"] if first_halt else None,
        "t0": first_halt["t0"] if first_halt else None,
        "t1": first_halt["t1"] if first_halt else None,
        "reason": first_halt["reason"] if first_halt else (history[-1]["text"] if history else ""),
        "p_halt": first_halt["p_halt"] if first_halt else (history[-1]["p_halt"] if history else None),
        "n_chunks": len(chunks),
        "duration": duration,
        "trace": history,
    }


def score_row(split: str, res: dict, gt_halt: dict | None) -> dict:
    """Streaming-oriented scoring for one video."""
    fh = res.get("first_halt")
    out = {
        "binary_ok": False,
        "false_halt": False,
        "missed_halt": False,
        "early_halt": False,
        "on_time_halt": False,
        "late_halt": False,
        "halt_latency_s": None,
    }
    if split == "correct":
        # expect never halt
        out["binary_ok"] = fh is None
        out["false_halt"] = fh is not None
        if fh is not None:
            out["halt_latency_s"] = float(fh["t0"])
        return out

    # mistake: expect halt
    if fh is None:
        out["missed_halt"] = True
        out["binary_ok"] = False
        return out

    out["halt_latency_s"] = float(fh["t0"])
    if gt_halt is None:
        out["binary_ok"] = True  # any halt counts if no GT window
        return out

    # Use halt chunk center vs GT window
    t_mid = 0.5 * (fh["t0"] + fh["t1"])
    g0, g1 = float(gt_halt["t0"]), float(gt_halt["t1"])
    if t_mid < g0:
        out["early_halt"] = True
        out["binary_ok"] = False  # streaming: early stop before error is wrong
    elif t_mid > g1:
        out["late_halt"] = True
        out["binary_ok"] = True  # still caught, but late
        out["on_time_halt"] = False
    else:
        out["on_time_halt"] = True
        out["binary_ok"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/scratch/ll5914/Labos/FineBioStreaming/outputs/streaming_vlm_v1")
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-NeXT-Video-7B-DPO")
    ap.add_argument("--out-dir", default="/scratch/ll5914/Labos/FineBioStreaming/outputs/eval_streaming_v2")
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=20.0)
    ap.add_argument("--halt-threshold", type=float, default=0.45)
    ap.add_argument("--decision", default="lm", choices=["lm", "head", "agree", "either"])
    ap.add_argument("--also-compare-modes", action="store_true",
                    help="After full traces, re-score all decision modes offline")
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    ckpt = Path(args.checkpoint)
    scfg = {}
    if (ckpt / "config_streaming.json").exists():
        scfg = json.loads((ckpt / "config_streaming.json").read_text())
    base = scfg.get("model_path", args.model_path)
    num_frames = scfg.get("num_frames", args.num_frames)
    pool_stride = scfg.get("pool_stride", args.pool_stride)

    overwrite = build_overwrite_config(base, num_frames, pool_stride)
    model_name = get_model_name_from_path(base)
    print(f"[eval] loading {base} + {ckpt}", flush=True)
    print(f"[eval] decision={args.decision} window={args.window_sec}s threshold={args.halt_threshold}", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        base, None, model_name, torch_dtype="bfloat16",
        overwrite_config=overwrite, attn_implementation=args.attn_impl,
    )
    if (ckpt / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt))
    halt_head = None
    aux = ckpt / "aux_heads.bin"
    if aux.exists():
        blob = torch.load(aux, map_location="cpu", weights_only=False)
        halt_head = nn.Linear(model.config.hidden_size, blob["n_halt"])
        halt_head.load_state_dict(blob["halt_head"])
        halt_head = halt_head.to(model.device, dtype=torch.bfloat16).eval()
    model.eval()

    rows = []
    for case in CASES:
        print("=" * 70, flush=True)
        print(f"[case] {case['name']} proto={case['proto']} ({case['note']})", flush=True)
        for split, video, expect in (
            ("correct", case["correct"], "CONTINUE"),
            ("mistake", case["mistake"], "HALT"),
        ):
            print(f"\n--- {case['name']} / {split} ---", flush=True)
            res = run_stream(
                model, tokenizer, image_processor, halt_head,
                video, case["proto"], num_frames, args.window_sec,
                args.halt_threshold, args.conv_mode, args.decision,
            )
            metrics = score_row(split, res, case.get("gt_halt") if split == "mistake" else None)
            row = {
                "case": case["name"],
                "split": split,
                "expect": expect,
                "got": res["verdict"],
                "note": case["note"],
                "error_type": case["error_type"],
                "gt_halt": case.get("gt_halt"),
                "halt_chunk": res.get("halt_chunk"),
                "t0": res.get("t0"),
                "t1": res.get("t1"),
                "p_halt_at_first": res.get("p_halt"),
                "n_chunks": res["n_chunks"],
                "duration": res["duration"],
                "reason": str(res.get("reason", ""))[:300],
                **metrics,
            }
            rows.append(row)
            (out_dir / f"{case['name']}_{split}.json").write_text(json.dumps(res, indent=2))
            print(
                f"[{split}] got={res['verdict']} halt_chunk={res.get('halt_chunk')} "
                f"t={res.get('t0')}-{res.get('t1')} binary_ok={metrics['binary_ok']} "
                f"false_halt={metrics['false_halt']} miss={metrics['missed_halt']} "
                f"early={metrics['early_halt']} on_time={metrics['on_time_halt']} late={metrics['late_halt']}",
                flush=True,
            )
            print(f"  reason: {str(res.get('reason', ''))[:200]}", flush=True)
            # brief head vs lm disagreement count
            n_disagree = sum(
                1 for h in res["trace"]
                if h["text_halt"] != bool(h["p_halt"] is not None and h["p_halt"] >= args.halt_threshold)
            )
            print(f"  chunks={res['n_chunks']} lm/head_disagree={n_disagree}", flush=True)

    # Offline re-score alternate decision modes from saved full traces
    mode_scores = {}
    if args.also_compare_modes:
        for mode in ["lm", "head", "agree", "either"]:
            mode_rows = []
            for case in CASES:
                for split in ("correct", "mistake"):
                    trace_path = out_dir / f"{case['name']}_{split}.json"
                    full = json.loads(trace_path.read_text())
                    first = None
                    for h in full["trace"]:
                        text_halt = bool(h["text_halt"])
                        halted = decide_halt(text_halt, h.get("p_halt"), args.halt_threshold, mode)
                        if halted and first is None:
                            first = {"chunk": h["chunk"], "t0": h["t0"], "t1": h["t1"],
                                     "reason": h["text"], "p_halt": h.get("p_halt")}
                            break
                    fake = {"first_halt": first}
                    m = score_row(split, fake, case.get("gt_halt") if split == "mistake" else None)
                    mode_rows.append(m)
            mode_scores[mode] = {
                "binary_ok": sum(1 for r in mode_rows if r["binary_ok"]),
                "max": len(mode_rows),
                "false_halts": sum(1 for r in mode_rows if r["false_halt"]),
                "missed": sum(1 for r in mode_rows if r["missed_halt"]),
                "early": sum(1 for r in mode_rows if r["early_halt"]),
                "on_time": sum(1 for r in mode_rows if r["on_time_halt"]),
                "late": sum(1 for r in mode_rows if r["late_halt"]),
            }

    binary_ok = sum(1 for r in rows if r["binary_ok"])
    report = {
        "checkpoint": str(ckpt),
        "decision": args.decision,
        "halt_threshold": args.halt_threshold,
        "window_sec": args.window_sec,
        "chunking": "fixed_window_all_splits",
        "binary_ok": binary_ok,
        "max_points": len(rows),
        "accuracy": binary_ok / max(len(rows), 1),
        "false_halts": sum(1 for r in rows if r["false_halt"]),
        "missed_halts": sum(1 for r in rows if r["missed_halt"]),
        "early_halts": sum(1 for r in rows if r["early_halt"]),
        "on_time_halts": sum(1 for r in rows if r["on_time_halt"]),
        "late_halts": sum(1 for r in rows if r["late_halt"]),
        "mode_compare": mode_scores,
        "rows": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# FineBioStreaming eval v2 (streaming metrics)",
        "",
        f"**Checkpoint:** `{ckpt}`  ",
        f"**Decision:** `{args.decision}`  ",
        f"**Chunking:** fixed {args.window_sec}s windows (correct & mistake)  ",
        f"**Streaming binary:** **{binary_ok}/{len(rows)}** "
        f"({100 * binary_ok / max(len(rows), 1):.0f}%)  ",
        f"**halt_threshold:** {args.halt_threshold}",
        "",
        "## Streaming summary",
        "",
        f"- false_halt (correct stopped early): **{report['false_halts']}**",
        f"- missed_halt (mistake never stopped): **{report['missed_halts']}**",
        f"- early_halt (before GT window): **{report['early_halts']}**",
        f"- on_time_halt (inside GT window): **{report['on_time_halts']}**",
        f"- late_halt (after GT window): **{report['late_halts']}**",
        "",
    ]
    if mode_scores:
        md += ["## Offline decision-mode compare", "",
               "| Mode | binary_ok | false | miss | early | on_time | late |",
               "|------|-----------|-------|------|-------|---------|------|"]
        for mode, s in mode_scores.items():
            md.append(
                f"| {mode} | {s['binary_ok']}/{s['max']} | {s['false_halts']} | "
                f"{s['missed']} | {s['early']} | {s['on_time']} | {s['late']} |"
            )
        md.append("")

    md += [
        "## Per video",
        "",
        "| Case | Split | Expect | Got | OK | Halt chunk | t (s) | Flags |",
        "|------|-------|--------|-----|----|------------|-------|-------|",
    ]
    for r in rows:
        flags = []
        if r["false_halt"]:
            flags.append("false_halt")
        if r["missed_halt"]:
            flags.append("miss")
        if r["early_halt"]:
            flags.append("early")
        if r["on_time_halt"]:
            flags.append("on_time")
        if r["late_halt"]:
            flags.append("late")
        tspan = f"{r['t0']:.1f}-{r['t1']:.1f}" if r.get("t0") is not None else "-"
        md.append(
            f"| {r['case']} | {r['split']} | {r['expect']} | {r['got']} | "
            f"{'Y' if r['binary_ok'] else 'N'} | {r['halt_chunk']} | {tspan} | "
            f"{','.join(flags) or '-'} |"
        )
    md.append("")
    md.append("## GT halt windows (approx)")
    for c in CASES:
        g = c["gt_halt"]
        md.append(f"- **{c['name']}** ({c['note']}): [{g['t0']:.0f}, {g['t1']:.0f}]s")
    md.append("")
    md.append("## Reasons (at first halt / final)")
    for r in rows:
        md.append(f"### {r['case']} / {r['split']}")
        md.append(f"- `{r['reason']}`")
        md.append("")

    (out_dir / "report.md").write_text("\n".join(md))
    print("=" * 70, flush=True)
    print(f"[done] {binary_ok}/{len(rows)} -> {out_dir / 'report.md'}", flush=True)
    if mode_scores:
        print(f"[modes] {json.dumps(mode_scores)}", flush=True)


if __name__ == "__main__":
    main()
