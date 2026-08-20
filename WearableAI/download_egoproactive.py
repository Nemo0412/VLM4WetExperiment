#!/usr/bin/env python3
"""Download EgoProactive videos + jsonl into scratch (HTTP, no Xet)."""
from __future__ import annotations

import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

token = os.environ.get("HF_TOKEN") or open("/home/ll5914/.hf_token").read().strip()
print("downloading egoproactive/**", flush=True)
p = snapshot_download(
    repo_id="facebook/wearable-ai",
    repo_type="dataset",
    token=token,
    local_dir="/scratch/ll5914/datasets/wearable-ai",
    allow_patterns=["egoproactive/**"],
    max_workers=8,
)
print("done", p, flush=True)
