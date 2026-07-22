# FineBioQwenStream

Protocol-level stream monitor with **Qwen2.5-VL-7B-Instruct**: given plan \(P_1{\to}P_2{\to}\cdots\) + video prefix → `CONTINUE` or `HALT`.

## Task

| | |
|--|--|
| Positive | \(P_1\), \(P_1{+}P_2\), … → `CONTINUE` |
| Negative | skip / wrong protocol clip → `HALT` |

Error types (2): `missing_protocol` | `wrong_execution`. HALT target: first ≤5 frames of the bad segment.

## Current experiments (running)

| Exp | What | Status / notes |
|--|--|--|
| **Zeroshot FPS ablation** | Base Qwen on val; grid = `stored`, 0.25, 0.5, 1, 2, 4 FPS | Job queues behind GPU; results → `outputs/fps_ablation_zeroshot/` |
| **LoRA SFT v2** | After ablation; `max_frames=8`, LoRA r=16, capped `max_pixels` | Prev v1 OOM’d (48 frames + default 12M pixels) |

**Sampling (v2 data):** 4 frames / protocol (uniform), concat capped at 8; HALT early-window still 1–5 native frames.

**OOM fixes vs v1:** `max_frames` 48→8, `max_pixels`≈128×28×28, LoRA r 64→16, last-layer hook (no full `hidden_states`).

## Loss

\[
L = w(k)\,(L_{\mathrm{lm}} + \lambda_{\mathrm{halt}} L_{\mathrm{halt}} + \lambda_{\mathrm{type}} L_{\mathrm{type}}),\quad
w(k)=1.5/k + 1{[k{=}5]}
\]

## Quickstart

```bash
# data (4 frames/proto)
python scripts/prepare_prefix_sft.py \
  --out-dir /scratch/$USER/Labos/FineBioQwenStream/data/proto_prefix_v2 \
  --frames-per-proto 4

# zeroshot FPS ablation only
sbatch run_eval_fps_ablation_a100.sbatch

# ablation + SFT (one GPU job)
sbatch run_zeroshot_then_train_a100.sbatch

# SFT only
sbatch run_train_a100.sbatch
```

Scratch outputs: `/scratch/$USER/Labos/FineBioQwenStream/outputs/`.

## Layout

| Path | Role |
|--|--|
| `protocol_prompt.py` | CONTINUE/HALT prompts + 2 error types |
| `scripts/prepare_prefix_sft.py` | prefix SFT jsonl |
| `train_sft.py` | LoRA SFT |
| `eval_prefix.py` | val metrics + `--fps-grid` ablation |
| `infer_stream.py` | single-clip infer |
| `run_*.sbatch` | Slurm |
