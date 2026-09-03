"""LabGate streaming window: reprojection → small Judger VLM → optional large VLM."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FineBioWhen2See"))
from gate import select_indices  # noqa: E402

from asr import audio_to_text
from models import QwenVL
from prompts import expert_prompt, judger_prompt, parse_expert, parse_judger


def load_indices(video: str, indices: list[int]) -> np.ndarray:
    vr = VideoReader(video, ctx=cpu(0), num_threads=2)
    n = len(vr)
    idxs = [min(max(int(i), 0), n - 1) for i in indices] or [0]
    return vr.get_batch(idxs).asnumpy()


def sample_window(video: str, t0: float, t1: float, n_all: int = 16) -> list[int]:
    vr = VideoReader(video, ctx=cpu(0), num_threads=2)
    fps = float(vr.get_avg_fps() or 30.0)
    n = len(vr)
    i0 = max(0, int(round(t0 * fps)))
    i1 = min(n - 1, int(round((t1 if t1 > t0 else t0 + 1.0) * fps)))
    span = max(1, i1 - i0 + 1)
    k = min(n_all, span)
    mid = (i0 + i1) // 2
    start = max(i0, min(mid - k // 2, i1 - k + 1))
    return list(range(start, start + k))


def reproject_keep(
    frames: np.ndarray, tau: float = 0.12
) -> tuple[np.ndarray, list[int]]:
    seq = [frames[i] for i in range(len(frames))]
    keep = select_indices(seq, mode="reproj", tau=tau)
    return np.stack([seq[i] for i in keep], axis=0), keep


@dataclass
class GateResult:
    fired: bool
    output_type: str
    message: str
    judger_raw: str
    expert_raw: str
    judger_reason: Optional[str]
    n_frames_in: int
    n_frames_kept: int
    keep_local: list[int]
    latency_judger_s: float
    latency_expert_s: float
    latency_e2e_s: float

    def to_dict(self):
        return asdict(self)


class LabGate:
    """Audio2Text + reprojection + Judger → optional large VLM."""

    def __init__(self, judger: QwenVL, expert: QwenVL, tau_reproj: float = 0.12):
        self.judger = judger
        self.expert = expert
        self.tau_reproj = tau_reproj

    def run_frames(
        self,
        frames: np.ndarray,
        protocol: str,
        asr_text: str = "",
        always_expert: bool = False,
    ) -> GateResult:
        kept, keep_local = reproject_keep(frames, tau=self.tau_reproj)
        dt_j = 0.0
        j_reply = ""
        reason = None
        if always_expert:
            fire = True
        else:
            j_reply, dt_j = self.judger.generate(
                kept, judger_prompt(protocol, asr_text)
            )
            fire, reason = parse_judger(j_reply)

        dt_e = 0.0
        e_reply = ""
        output_type, message = "NONE", ""
        if fire:
            e_reply, dt_e = self.expert.generate(
                kept, expert_prompt(protocol, asr_text)
            )
            output_type, message = parse_expert(e_reply)

        return GateResult(
            fired=fire,
            output_type=output_type if fire else "NONE",
            message=message if fire else "",
            judger_raw=j_reply,
            expert_raw=e_reply,
            judger_reason=reason,
            n_frames_in=int(len(frames)),
            n_frames_kept=int(len(kept)),
            keep_local=list(keep_local),
            latency_judger_s=dt_j,
            latency_expert_s=dt_e,
            latency_e2e_s=dt_j + dt_e,
        )

    def run_clip(
        self,
        video: str,
        protocol: str,
        indices: list[int] | None = None,
        t0: float = 0.0,
        t1: float = 0.0,
        asr_text: str | None = None,
        audio_path: str | None = None,
        n_all: int = 16,
        always_expert: bool = False,
    ) -> GateResult:
        speech = audio_to_text(asr_text=asr_text, audio_path=audio_path)
        if not indices:
            indices = sample_window(video, t0, t1, n_all=n_all)
        frames = load_indices(video, indices)
        return self.run_frames(
            frames, protocol, speech, always_expert=always_expert
        )
