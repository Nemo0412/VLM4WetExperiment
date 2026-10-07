#!/usr/bin/env python3
"""Light matmul so an allocated H200 is not idle during the weight download."""
import time

import torch

n = torch.cuda.device_count()
if n < 1:
    raise SystemExit("keepalive: no CUDA device")
tensors = [torch.randn(2048, 2048, device=f"cuda:{i}") for i in range(n)]
while True:
    for i, a in enumerate(tensors):
        b = a @ a
        tensors[i] = b / (b.norm() + 1e-6)
    torch.cuda.synchronize()
    time.sleep(0.05)
