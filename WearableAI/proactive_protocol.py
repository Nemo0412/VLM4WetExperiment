"""Wearable-AI EgoProactive protocol helpers (official starter_kit alignment)."""

from __future__ import annotations

import os
import sys

# Official frame extraction lives in the HF starter kit.
STARTER_KIT = "/scratch/ll5914/datasets/wearable-ai/starter_kit"
if STARTER_KIT not in sys.path:
    sys.path.insert(0, STARTER_KIT)

from model import extract_frames  # noqa: E402

SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the "
    "user performing a procedural task. The user has issued a single "
    "high-level query. As the video unfolds you observe a series of short "
    "(~8s) chunks; after each chunk you decide whether to speak or stay "
    "silent.\n\n"
    "Output format (single line, no preamble):\n"
    "  - If you should speak: start with the literal token `$interrupt$` "
    "followed by your suggestion or answer in plain text.\n"
    "  - If you should stay silent: output the single literal token "
    "`$silent$` and nothing else.\n\n"
    "Speak when the user asks you something, when an earlier action needs "
    "correction, or when you have useful, timely guidance for the next step. "
    "Stay silent when nothing useful needs to be said."
)


def normalize_dialog_turns(dialog_at_chunk: list[dict]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for turn in dialog_at_chunk:
        text = turn.get("text") or ""
        if not text:
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        history.append({"role": role, "content": str(text)})
    return history


def build_messages(
    query: str,
    dialog_at_chunk: list[dict],
    *,
    max_history_turns: int = 4,
) -> list[dict[str, str]]:
    """Build chat messages for one chunk (matches run_generate_proactive.py)."""
    turns_after_query = dialog_at_chunk[1:] if len(dialog_at_chunk) >= 1 else []
    if max_history_turns == 0:
        turns_after_query = []
    elif max_history_turns > 0:
        turns_after_query = turns_after_query[-max_history_turns:]
    history = normalize_dialog_turns(turns_after_query)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if query:
        messages.append({"role": "user", "content": query})
    messages.extend(history)
    return messages


def extract_cumulative_frames(
    video_path: str,
    intervals: list[list[float]],
    chunk_index: int,
    *,
    frames_per_interval: int = 16,
    max_frames: int = 32,
) -> list[object]:
    """Cumulative frames for chunks 0..chunk_index (official proactive default)."""
    all_intervals = [(float(s), float(e)) for s, e in intervals]
    frames_per_chunk: list[list[object]] = []
    for interval in all_intervals:
        frames_per_chunk.append(
            extract_frames(
                video_path,
                intervals=[interval],
                frames_per_interval=frames_per_interval,
            )
        )

    frames: list[object] = []
    for k in range(chunk_index + 1):
        frames.extend(frames_per_chunk[k])
    if max_frames > 0 and len(frames) > max_frames:
        stride = len(frames) / max_frames
        frames = [frames[int(idx * stride)] for idx in range(max_frames)]
    return frames


def video_file(video_folder: str, video_path: str) -> str:
    return os.path.join(video_folder, video_path)


def parse_decision(answer: str) -> tuple[str, str]:
    text = (answer or "").strip()
    if text.lower().startswith("$silent$"):
        return "Silent", ""
    if text.lower().startswith("$interrupt$"):
        return "Interrupt", text[len("$interrupt$") :].strip()
    return "Unknown", text
