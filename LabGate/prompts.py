"""Zero-shot prompts for the small Judger and the large expert VLM."""

from __future__ import annotations

import re

OUTPUT_TYPES = ("SAFETY", "ACTION_ERROR", "ASSISTANT", "NONE")


def judger_prompt(protocol: str, asr_text: str) -> str:
    speech = asr_text.strip() if asr_text and asr_text.strip() else "(no user speech)"
    return (
        "You are a fast lab-monitor gate. Decide whether the LARGE assistant VLM "
        "must be called on this moment.\n\n"
        f"PROTOCOL:\n{protocol}\n\n"
        f"USER SPEECH (ASR):\n{speech}\n\n"
        "Call YES if ANY of these is true:\n"
        "  1. Safety issue (hazard, spill, centrifuge lid open, fire, injury, ethanol near heat).\n"
        "  2. User asked a question or requested help (ASR is a question / request).\n"
        "  3. Visible actions conflict with the protocol: wrong experiment for this protocol, "
        "skipped required step, extra wash, wrong instrument, or ASR mentions a mistake.\n"
        "When unsure about a protocol conflict, prefer YES so the large model can check.\n"
        "Otherwise NO. Routine correct work with silence → NO.\n\n"
        "Reply with EXACTLY one line:\n"
        "YES. reason=<safety|action_error|user_query>\n"
        "or\n"
        "NO.\n"
    )


def expert_prompt(protocol: str, asr_text: str) -> str:
    speech = asr_text.strip() if asr_text and asr_text.strip() else "(no user speech)"
    return (
        "You are a wet-lab copilot watching first-person video plus the protocol.\n\n"
        f"PROTOCOL:\n{protocol}\n\n"
        f"USER SPEECH (ASR):\n{speech}\n\n"
        "Produce exactly one decision:\n"
        "  SAFETY       — hazard / unsafe handling. Warn immediately.\n"
        "  ACTION_ERROR — what the person is doing conflicts with the protocol.\n"
        "  ASSISTANT    — user asked something; answer using protocol + video.\n"
        "  NONE         — nothing to report.\n\n"
        "Reply in EXACTLY this format (two lines):\n"
        "TYPE: <SAFETY|ACTION_ERROR|ASSISTANT|NONE>\n"
        "MSG: <one or two sentences>\n"
    )


def parse_judger(text: str) -> tuple[bool, str | None]:
    t = (text or "").strip()
    up = t.upper()
    if up.startswith("NO"):
        return False, None
    if "YES" in up:
        match = re.search(r"reason\s*=\s*([a-z_]+)", t, flags=re.I)
        return True, match.group(1).lower() if match else None
    if re.search(r"\byes\b", t, flags=re.I):
        return True, None
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
