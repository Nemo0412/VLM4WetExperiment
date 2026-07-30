# FineBioQwenStream

Protocol-level stream monitor with **Qwen2.5-VL-7B-Instruct**: given plan \(P_1{\to}P_2{\to}\cdots\) + video prefix → `CONTINUE` or `HALT`.

## Task

| | |
|--|--|
| Positive | \(P_1\), \(P_1{+}P_2\), … → `CONTINUE` |
| Negative | skip / wrong protocol clip → `HALT` |

Error types (2): `missing_protocol` | `wrong_execution`. HALT target: first ≤5 frames of the bad segment. Reason text is **not** scored.

## Results (ckpt-1836, proto_prefix_v2)

| Stage | Decision | et@HALT / notes |
|--|--|--|
| Zeroshot FPS ablation | ~5.5% | et@HALT 0% (always CONTINUE) |
| Fixed-index SFT eval | val 91.3% / test 92.1% | 76.7% / 77.7% |
| Natural streaming (≤5) | test detect 99.0% / val 96.9% | CONTINUE false-HALT still ~21–32% |

**Caveat:** current prefixes concat clips across subjects; same-subject streaming (v3) is planned, not implemented.

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

# optional frame cache (helps GPU util / Slurm kill protection)
python scripts/prepare_frame_cache.py ...

# zeroshot FPS ablation only
sbatch run_eval_fps_ablation_a100.sbatch

# ablation + SFT (one GPU job; auto-resubmit until TRAINING_DONE)
sbatch run_zeroshot_then_train_a100.sbatch

# SFT only
sbatch run_train_a100.sbatch

# natural streaming eval
sbatch run_eval_natural_stream_a100.sbatch
```

Scratch outputs: `/scratch/$USER/Labos/FineBioQwenStream/outputs/`.

## Layout

| Path | Role |
|--|--|
| `protocol_prompt.py` | CONTINUE/HALT prompts + 2 error types |
| `scripts/prepare_prefix_sft.py` | prefix SFT jsonl |
| `scripts/prepare_frame_cache.py` | decode frames for fast train |
| `frame_cache.py` | cache loader |
| `train_sft.py` | LoRA SFT + resume / USR1 |
| `eval_prefix.py` | val metrics + `--fps-grid` ablation |
| `eval_stream_natural.py` | natural FPS rollout eval |
| `infer_stream.py` | single-clip infer |
| `miss_case_frames/` | example failure frames |
| `run_*.sbatch` | Slurm |
