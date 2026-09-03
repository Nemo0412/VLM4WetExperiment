# LabGate

Zero-shot wet-lab monitor with a cheap frame gate, a small Judger VLM, and a
large expert VLM:

```text
Audio ──► Audio2Text ──────────────────────┐
Video ──► reprojection ─► Judger (Qwen2.5-VL-3B) ─YES─► Expert (32B)
Protocol ──────────────────────────────────┘                 │
                                                SAFETY / ACTION_ERROR / ASSISTANT
```

The Judger only calls the 32B expert for safety issues, user questions, or
protocol conflicts. `FineBioWhen2See/gate.py` removes frames explained by
viewpoint reprojection.

## Zero-shot accuracy test

`eval_zeroshot.py` is the accuracy/latency script. It evaluates 42 FineBio
cases across `none`, `assistant`, `safety`, and `action_error`, comparing the
3B→32B gate with always running 32B.

On the NYU cluster, run:

```bash
cd LabGate
sbatch run_zeroshot_32b_h200.sbatch
```

The job builds the cases with `prepare_eval.py`, then runs:

```bash
python eval_zeroshot.py \
  --eval-jsonl /scratch/ll5914/Labos/LabGate/data/eval_v2.jsonl \
  --judger Qwen/Qwen2.5-VL-3B-Instruct \
  --expert Qwen/Qwen2.5-VL-32B-Instruct \
  --always-expert \
  --out /scratch/ll5914/Labos/LabGate/outputs/zeroshot_3b32b.json
```

Measured zero-shot result: **76.2%** end-to-end type accuracy, **0.82** gate
F1, and **0.93 s** gated latency versus **1.45 s** always-32B latency.

Run one clip:

```bash
python run_one.py \
  --video /path/to/video.mp4 \
  --protocol-id 3 \
  --asr "What should I do next?" \
  --t0 20 --t1 28
```

FineBio videos, model weights, predictions, and logs are intentionally excluded
from Git.
