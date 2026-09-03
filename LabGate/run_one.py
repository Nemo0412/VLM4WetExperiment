#!/usr/bin/env python3
"""Run LabGate on one clip."""

from __future__ import annotations

import argparse
import json

from models import QwenVL
from pipeline import LabGate
from protocols import format_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--protocol-id", type=int, default=3)
    parser.add_argument("--asr", default="")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=8.0)
    parser.add_argument("--judger", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--expert", default="Qwen/Qwen2.5-VL-32B-Instruct")
    args = parser.parse_args()

    gate = LabGate(
        QwenVL(args.judger, max_new_tokens=24),
        QwenVL(args.expert, max_new_tokens=96),
    )
    result = gate.run_clip(
        args.video,
        format_protocol(args.protocol_id),
        t0=args.t0,
        t1=args.t1,
        asr_text=args.asr,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
