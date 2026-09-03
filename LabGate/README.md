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

## Evaluation

The four labels are:

- `none`: correct trial and no speech
- `assistant`: a user question supplied as ASR text
- `safety`: a spoken hazard supplied as ASR text
- `action_error`: FineBio mistakes, spoken mistakes, and video/protocol mismatch

Metrics include end-to-end type accuracy, Judger precision/recall/F1, per-type
accuracy, retained frames, and gated versus always-expert latency.

```bash
cd LabGate
python prepare_eval.py --out /scratch/$USER/Labos/LabGate/data/eval_v2.jsonl
sbatch run_zeroshot_32b_h200.sbatch
```

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
