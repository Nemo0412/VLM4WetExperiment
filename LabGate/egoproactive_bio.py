"""Label-independent prompts, strict gate parsing, and metrics for EgoProactive-Bio."""

from __future__ import annotations

import math
import re


# Static SOP distilled from source_docs/cell_passaging_protocol_and_action_v2.docx.
# Deliberately excludes observed-action times, error clip IDs, and decision labels.
CELL_PASSAGING_PROTOCOL = """Cell passaging (provided experimental protocol):
1. Sanitize hands. Take the cell flask from the incubator, disinfect its exterior,
   and place it in the biosafety cabinet. UV must be off, the sash at the
   recommended working height, and hands gloved and disinfected for cabinet work.
2. Remove spent medium, promptly rinse cells with PBS, then discard the PBS.
   Do not leave the flask dry and exposed after removing the medium.
3. Add trypsin. Gently rock or tap to distribute it; avoid vigorous shaking.
   Disinfect the flask, place it in the incubator, and incubate for 2 minutes.
4. Add the equivalent of 2 volumes of complete growth medium to neutralize trypsin.
   Do not prolong trypsin exposure. Gently pipette to disperse the cells.
5. Transfer the appropriate volume to a new flask, add fresh culture medium,
   label the new flask, and return it to the incubator.
"""


def parse_gate(text: str) -> str:
    """Accept an initial decision token, never a YES mentioned inside an explanation."""
    match = re.match(
        r"^\s*(?:\$(interrupt|silent)\$|(yes|no|interrupt|interupt|silent)\b)",
        text or "", re.I,
    )
    if not match:
        return "invalid"
    token = (match.group(1) or match.group(2)).lower()
    return "interrupt" if token in {"yes", "interrupt", "interupt"} else "silent"


def context_text(query, history, timestamps, decision_time):
    dialog = "\n".join(f"{turn['role']}: {turn['text']}" for turn in history)
    return (
        f"TASK: {query}\nPROTOCOL:\n{CELL_PASSAGING_PROTOCOL}\n"
        f"PREVIOUS DIALOG (before this decision):\n{dialog}\n"
        f"Current video time: {decision_time:.3f} seconds.\n"
        "Frames are a sparse chronological video prefix followed by the current window. "
        "Only frames before the current decision time are shown; gaps are not actions. "
        "This video contains editing cuts. Do not infer an unseen error from a cut alone.\n"
        "Frame timestamps in seconds: " + ", ".join(f"{t:.3f}" for t in timestamps)
        + "\nThere is no audio or new user question in this sample.\n"
    )


def gate_prompt(context):
    return (
        "You are the small decision VLM in a proactive lab assistant. Decide whether "
        "to interrupt NOW. Do not generate guidance.\n" + context
        + "Interrupt if an immediate visible safety/protocol error needs correction, "
        "or a completed step makes a brief next-step instruction useful now. "
        "Stay silent during ongoing correct work, after a correction is being followed, "
        "when the next action has already been explained, or after the task is finished. "
        "Judge the current moment, not an earlier error that has since been corrected. "
        "Do not claim an error solely because a required step was not sampled.\n"
        "Reply with exactly one token: $interrupt$ or $silent$."
    )


def guidance_prompt(context):
    return (
        "You are the large guidance VLM. The small VLM has requested an interruption.\n"
        + context
        + "Produce a concise spoken instruction in English (one or two sentences) "
        "grounded in the visible current action and supplied protocol. Correct a visible "
        "error or provide the appropriate next step. Do not repeat resolved warnings, "
        "invent observations, or add unsupported quantities. If the visual evidence is "
        "unclear, phrase the instruction conditionally. Output only the guidance text, "
        "without labels, analysis, or a silent/interrupt decision."
    )


def sample_indices(start, end, fps, n_video, window_frames=8, prefix_frames=8):
    """Uniform window plus sparse prefix, with strictly causal half-open endpoints."""
    if not (0 <= start < end and fps > 0 and n_video > 0 and window_frames > 0
            and prefix_frames >= 0):
        raise ValueError("Invalid interval, video metadata, or frame budget")
    lo = min(n_video - 1, max(0, math.ceil(start * fps)))
    hi = min(n_video - 1, math.ceil(end * fps) - 1)
    if hi < lo:
        raise ValueError("No video frame in decision window")

    def uniform(a, b, n):
        n = min(n, b - a + 1)
        if n <= 0:
            return []
        if n == 1:
            return [b]
        return [round(a + i * (b - a) / (n - 1)) for i in range(n)]

    return sorted(set(uniform(0, lo - 1, prefix_frames)
                      + uniform(lo, hi, window_frames)))


def run_gated(judger, expert, frames, context):
    """Expert generation is reachable ONLY for a positive small-model decision."""
    j_prompt = gate_prompt(context)
    raw, j_time = judger.generate(frames, j_prompt)
    prediction = parse_gate(raw)
    fired = prediction == "interrupt"
    expert_raw, e_time, e_prompt = "", 0.0, ""
    if fired:
        e_prompt = guidance_prompt(context)
        expert_raw, e_time = expert.generate(frames, e_prompt)
    guidance = re.sub(r"^\s*\$interrupt\$\s*", "", expert_raw).strip()
    valid_guidance = bool(guidance) and parse_gate(guidance) != "silent"
    return {
        "pred_label": prediction, "fired": fired, "judger_raw": raw,
        "expert_raw": expert_raw, "guidance": guidance,
        "guidance_nonempty": fired and valid_guidance,
        "answer": "$interrupt$" + guidance if fired else "$" + prediction + "$",
        "judger_prompt": j_prompt, "expert_prompt": e_prompt,
        "latency_judger_generate_s": j_time, "latency_expert_generate_s": e_time,
    }


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot score zero decisions")
    classes = {}
    for label in ("interrupt", "silent"):
        tp = sum(r['gt_label'] == label and r['pred_label'] == label for r in rows)
        fp = sum(r['gt_label'] != label and r['pred_label'] == label for r in rows)
        fn = sum(r['gt_label'] == label and r['pred_label'] != label for r in rows)
        classes[label] = {
            "support": sum(r['gt_label'] == label for r in rows),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    confusion = {
        gt: {pred: sum(r['gt_label'] == gt and r['pred_label'] == pred for r in rows)
             for pred in ("interrupt", "silent", "invalid")}
        for gt in ("interrupt", "silent")
    }
    mean = lambda values: sum(values) / len(values) if values else None
    fired = [r for r in rows if r['fired']]
    groups = {}
    for phase in sorted({r['phase'] for r in rows}):
        group = [r for r in rows if r['phase'] == phase]
        groups[phase] = {
            "n": len(group), "correct": sum(r['gt_label'] == r['pred_label'] for r in group),
            "interrupts": sum(r['fired'] for r in group),
        }
    return {
        "n": n, "accuracy": sum(r['gt_label'] == r['pred_label'] for r in rows) / n,
        "macro_f1": sum(c['f1'] for c in classes.values()) / 2,
        "gmean_f1": math.sqrt(classes['interrupt']['f1'] * classes['silent']['f1']),
        "per_class": classes, "confusion_gt_pred": confusion,
        "invalid_outputs": sum(r['pred_label'] == 'invalid' for r in rows),
        "expert_calls": len(fired), "expert_call_rate": len(fired) / n,
        "expert_calls_avoided": n - len(fired),
        "guidance_nonempty_count": sum(r['guidance_nonempty'] for r in rows),
        "guidance_semantic_accuracy": None,
        "per_phase": groups,
        "mean_frames_in": mean([r['n_frames_in'] for r in rows]),
        "mean_frames_kept": mean([r['n_frames_kept'] for r in rows]),
        "latency_s": {
            "judger_generate_mean": mean([r['latency_judger_generate_s'] for r in rows]),
            "expert_generate_mean_when_fired": mean([r['latency_expert_generate_s'] for r in fired]),
            "decision_wall_mean": mean([r['latency_wall_s'] for r in rows]),
        },
    }
