"""Zero-shot prompts for the small Judger and the large expert VLM."""

from __future__ import annotations

import json
import re

OUTPUT_TYPES = ("SAFETY", "ACTION_ERROR", "ASSISTANT", "NONE")


def judger_prompt(
    protocol: str,
    asr_text: str,
    *,
    proactive_next_step: bool = False,
    current_step: str = "",
) -> str:
    speech = asr_text.strip() if asr_text and asr_text.strip() else "(no user speech)"
    step_context = (
        f"CURRENT TASK / STEP (provided context):\n{current_step.strip()}\n\n"
        if current_step and current_step.strip()
        else ""
    )
    next_step_rule = (
        "  4. A protocol step has visibly just completed and a timely next-step reminder "
        "would help; use reason=next_step.\n"
        if proactive_next_step
        else ""
    )
    yes_format = (
        "YES. reason=<safety|action_error|user_query|next_step>\n"
        if proactive_next_step
        else "YES. reason=<safety|action_error|user_query>\n"
    )
    history_timing_rule = (
        "History controls timing: if the current action is the correction or next action "
        "already requested by the latest assistant message, reply NO when it is being done "
        "correctly. Do not repeat an earlier instruction. Use reason=next_step only after the "
        "current step has visibly completed and the needed next action has not already been "
        "given in history. A provided recovery step being performed correctly is NO unless a "
        "new visible problem exists.\n"
        if proactive_next_step
        else ""
    )
    return (
        "You are a fast lab-monitor gate. Decide whether the LARGE assistant VLM "
        "must be called on this moment.\n\n"
        "Any conversation turns before the current video are HISTORY from earlier "
        "decisions. Never treat an earlier user turn as a new request now. The CURRENT "
        "TASK / STEP is supplied state context, not user speech or a request. Only the "
        "current USER SPEECH (ASR) field below may trigger reason=user_query; when it "
        "says (no user speech), reason=user_query is forbidden.\n\n"
        f"PROTOCOL:\n{protocol}\n\n"
        + step_context
        + f"USER SPEECH (ASR):\n{speech}\n\n"
        "Call YES if ANY of these is true:\n"
        "  1. Safety issue (hazard, spill, centrifuge lid open, fire, injury, ethanol near heat).\n"
        "  2. User asked a question or requested help (ASR is a question / request).\n"
        "  3. Visible actions conflict with the protocol: wrong experiment for this protocol, "
        "skipped required step, extra wash, wrong instrument, or ASR mentions a mistake.\n"
        + next_step_rule
        + history_timing_rule
        +
        "When unsure about a protocol conflict, prefer YES so the large model can check.\n"
        "Otherwise NO. Routine correct work with silence → NO.\n\n"
        "Reply with EXACTLY one line:\n"
        + yes_format
        +
        "or\n"
        "NO.\n"
    )


def trigger_context(reason_type: str | None, observation: str = "", current_step: str = "") -> str:
    """Pass model-produced trigger evidence explicitly, never as verified ground truth."""
    return (
        "SMALL VLM TRIGGER (unverified; check against the current visual evidence):\n"
        + json.dumps({"reason_type": reason_type or "unknown", "observation": observation,
                      "current_step": current_step}, ensure_ascii=False)
        + "\nDo not assume the proposed reason is correct or repeat a resolved historical warning.\n"
    )


def expert_prompt(
    protocol: str,
    asr_text: str,
    judger_reason: str | None = None,
    *,
    observation: str = "",
    current_step: str = "",
) -> str:
    speech = asr_text.strip() if asr_text and asr_text.strip() else "(no user speech)"
    step_context = (
        f"CURRENT TASK / STEP (provided context):\n{current_step.strip()}\n\n"
        if current_step and current_step.strip()
        else ""
    )
    return (
        "You are a wet-lab copilot watching first-person video plus the protocol.\n\n"
        f"PROTOCOL:\n{protocol}\n\n"
        + step_context
        + f"USER SPEECH (ASR):\n{speech}\n\n"
        + (
            trigger_context(judger_reason, observation, current_step)
            if judger_reason is not None
            else ""
        )
        +
        "Produce exactly one decision:\n"
        "  SAFETY       — hazard / unsafe handling. Warn immediately.\n"
        "  ACTION_ERROR — what the person is doing conflicts with the protocol.\n"
        "  ASSISTANT    — user asked something or the trigger reason is next_step; "
        "give timely guidance using protocol + video.\n"
        "  NONE         — nothing to report.\n\n"
        "Reply in EXACTLY this format (two lines):\n"
        "TYPE: <SAFETY|ACTION_ERROR|ASSISTANT|NONE>\n"
        "MSG: <one or two sentences>\n"
    )


def parse_judger(text: str) -> tuple[bool, str | None]:
    t = (text or "").strip()
    up = t.upper()
    if re.match(r"^NO\b", up):
        return False, None
    if re.match(r"^YES\b", up):
        match = re.search(r"reason\s*=\s*([a-z_]+)", t, flags=re.I)
        return True, match.group(1).lower() if match else None
    return False, None


def parse_expert(text: str) -> tuple[str, str]:
    t = (text or "").strip()
    match = re.search(
        r"TYPE\s*:\s*(SAFETY|ACTION_ERROR|ASSISTANT|NONE)", t, flags=re.I
    )
    output_type = match.group(1).upper() if match else "NONE"
    if output_type not in OUTPUT_TYPES:
        output_type = "NONE"
    message_match = re.search(r"MSG\s*:\s*(.+)", t, flags=re.I | re.S)
    message = message_match.group(1).strip() if message_match else t
    return output_type, message.splitlines()[0].strip()
