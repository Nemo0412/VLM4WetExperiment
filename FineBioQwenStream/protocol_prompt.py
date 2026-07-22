#!/usr/bin/env python3
"""Prompt / reply for *protocol-level* prefix streaming (no step labels).

P_i = entire FineBio trial video of protocol type i.
Intended stream: P1 → P2 → P3 → … (full protocol videos concatenated).

Mistake types (exactly 2):
  - missing_protocol : skipped / omitted a required protocol in the plan
  - wrong_execution  : current slot should be P_i but the video is a wrong experiment clip
"""

from __future__ import annotations

import re

PROTOCOL_NAMES = {
    1: "Cell lysate collection (single PBS wash)",
    2: "Cell lysate collection (double PBS wash)",
    3: "Magnetic-bead DNA extraction (single ethanol wash)",
    4: "Magnetic-bead DNA extraction (double ethanol wash)",
    5: "PCR reaction setup with 8-tube strips",
    6: "Spin-column DNA extraction (two wash steps)",
    7: "Spin-column DNA extraction (three wash steps)",
}

# Exactly two mistake classes
ERROR_TYPES = (
    "missing_protocol",  # 少了 xxx protocol（跳步 / 缺段）
    "wrong_execution",   # 当前该做的 protocol 做错了（插入离谱异源片段）
)

ERROR_TYPE_TO_ID = {e: i for i, e in enumerate(ERROR_TYPES)}


def format_intended_plan(protocol_ids: list[int]) -> str:
    lines = []
    for i, pid in enumerate(protocol_ids):
        name = PROTOCOL_NAMES.get(pid, f"protocol {pid}")
        lines.append(f"  P{i+1}. Protocol {pid}: {name}")
    return "\n".join(lines)


def build_user_prompt(intended: list[int], n_segments_seen: int) -> str:
    return (
        "You are monitoring a wet-lab session as a video stream.\n"
        "The experimenter must follow this ordered sequence of *full protocols* "
        "(each P_i is one entire protocol video):\n"
        f"{format_intended_plan(intended)}\n\n"
        f"You are watching a concatenation of the first {n_segments_seen} "
        "protocol video segment(s) seen so far.\n"
        "Decide whether the stream so far still follows the intended plan.\n\n"
        "Reply in EXACTLY one of these formats:\n"
        "  CONTINUE.\n"
        "or\n"
        "  HALT. error_type=<missing_protocol|wrong_execution>. <brief reason>\n"
        "error_type meanings:\n"
        "  missing_protocol  — a required protocol in the plan was skipped / is missing\n"
        "  wrong_execution   — the current protocol slot is being done incorrectly "
        "(wrong experiment footage)\n"
        "If a deviation just started, HALT immediately using the first frames of the bad segment."
    )


def build_assistant_reply(label: str, error_type: str = "", reason: str = "") -> str:
    if label == "continue":
        return "CONTINUE."
    et = error_type if error_type in ERROR_TYPE_TO_ID else "missing_protocol"
    reason = reason.strip() or "Stream diverges from the intended protocol sequence."
    return f"HALT. error_type={et}. {reason}"


def build_messages(
    intended: list[int],
    n_segments_seen: int,
    label: str,
    error_type: str = "",
    reason: str = "",
) -> list[dict]:
    return [
        {"role": "user", "content": build_user_prompt(intended, n_segments_seen)},
        {"role": "assistant", "content": build_assistant_reply(label, error_type, reason)},
    ]


_HALT_RE = re.compile(r"error_type\s*=\s*([a-z_]+)", re.I)


def parse_decision(text: str) -> tuple[str, str | None]:
    t = (text or "").strip()
    up = t.upper()
    if up.startswith("CONTINUE"):
        return "CONTINUE", None
    if "HALT" in up:
        m = _HALT_RE.search(t)
        et = m.group(1).lower() if m else None
        return "HALT", et
    return "UNKNOWN", None


def type_id(label: str, error_type: str | None) -> int:
    if label != "halt":
        return -100
    if not error_type or error_type not in ERROR_TYPE_TO_ID:
        return -100
    return ERROR_TYPE_TO_ID[error_type]
