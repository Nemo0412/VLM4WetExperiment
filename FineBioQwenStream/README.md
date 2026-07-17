# FineBioQwenStream

Protocol-level video streaming SFT with **Qwen2.5-VL** (no SSL, no step labels).

## Task

- Each \(P_i\) = one **full protocol** video.
- Input: intended plan \(P_1{\to}P_2{\to}\cdots\) + concatenated stream.
- Output: `CONTINUE` or `HALT. error_type=...`

| Positive | Negative |
|--|--|
| \(P_1\), \(P_1{+}P_2\), … | \(P_2\), \(P_1{+}P_3\), wrong clip inserted, … |

**Mistake types (2):**
- `missing_protocol` — required protocol skipped
- `wrong_execution` — current slot uses footage from another experiment

HALT must fire within the **first ≤5 frames** of the bad segment (prefer 1).

## Loss

\[
L = w(k)\,(L_{\mathrm{lm}} + \lambda_{\mathrm{halt}} L_{\mathrm{halt}} + \lambda_{\mathrm{type}} L_{\mathrm{type}})
\]

\(w(k)=1.5/k + 1{[k{=}5]}\) for halt horizons \(k{=}1..5\).

## Quickstart

```bash
# data
python scripts/prepare_prefix_sft.py \
  --out-dir /scratch/$USER/Labos/FineBioQwenStream/data/proto_prefix_v1

# train (1×A100)
sbatch run_train_a100.sbatch
```

## Layout

| Path | Role |
|--|--|
| `protocol_prompt.py` | prompts + 2-class types |
| `scripts/prepare_prefix_sft.py` | build pos/neg prefixes |
| `train_sft.py` | Qwen2.5-VL LoRA SFT |
| `infer_stream.py` / `eval_prefix.py` | inference / metrics |
| `run_train_a100.sbatch` | Slurm |
