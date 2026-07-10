#!/usr/bin/env python3
"""Build protocol-aware preference pairs from FineBio LLaVA samples.

Each pair is (preferred = intact compliance, rejected = corrupted compliance)
drawn from the *same trial*. This supports the ranking loss:

    prefer score(intact_grid) > score(corrupted_grid)

Scene-only samples are skipped (they have no natural negative counterpart).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def trial_key(sample_id: str) -> str:
    # P06_03_01_intact_comp / P06_03_01_synth0_comp -> P06_03_01
    m = re.match(r"(P\d+_\d+_\d+)_", sample_id)
    return m.group(1) if m else sample_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/scratch/ll5914/Labos/Llava/data/finebio_llava")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.data_dir)

    for split in ("train", "val"):
        data = json.loads((root / f"{split}.json").read_text())
        by_trial: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"good": [], "bad": []})
        for i, s in enumerate(data):
            if "_comp" not in s["id"]:
                continue
            bucket = "good" if int(s.get("integrity", -1)) == 1 else "bad"
            by_trial[trial_key(s["id"])][bucket].append(i)

        pairs = []
        for tid, buckets in by_trial.items():
            for g in buckets["good"]:
                for b in buckets["bad"]:
                    pairs.append({
                        "preferred_idx": g,
                        "rejected_idx": b,
                        "trial_id": tid,
                        "preferred_id": data[g]["id"],
                        "rejected_id": data[b]["id"],
                        "protocol_id": data[g]["protocol_id"],
                        "violation": (
                            "real_mistake" if "synth" not in data[b]["id"] else
                            ("missing" if "synth0" in data[b]["id"] or "synth2" in data[b]["id"]
                             else "reorder" if "synth1" in data[b]["id"] else "synthetic")
                        ),
                    })
        out = root / f"{split}_pairs.json"
        out.write_text(json.dumps(pairs, indent=2))
        n_trials = sum(1 for v in by_trial.values() if v["good"] and v["bad"])
        print(f"[{split}] samples={len(data)} pairable_trials={n_trials} pairs={len(pairs)} -> {out}")


if __name__ == "__main__":
    main()
