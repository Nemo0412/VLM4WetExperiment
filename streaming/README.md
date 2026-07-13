# FineBioStreaming

Video stream + protocol → **CONTINUE** / **HALT + reason** (stop on HALT).

Backbone: `LLaVA-NeXT-Video-7B-DPO` + LoRA + aux `halt_head` / `step_head`.

**v2:** prompt includes protocol progress history; HALT labeled at first error chunk; major mistakes time-aligned.

## Quickstart

```bash
# prepare chunk data (v2)
python scripts/prepare_streaming_data.py --out-dir /path/to/streaming_v2

# train (4× A100)
sbatch run_train_a100.sbatch

# stream one video
python infer_stream.py \
  --checkpoint /path/to/ckpt \
  --video /path/to/video.mp4 \
  --protocol-id 3 \
  --decision lm \
  --halt-threshold 0.45

# correct vs mistake (streaming metrics)
python eval_correct_mistake.py \
  --checkpoint /path/to/ckpt \
  --decision lm \
  --also-compare-modes
```

## Layout

| File | Role |
|------|------|
| `train_streaming.py` | LoRA + aux heads |
| `infer_stream.py` | chunk loop + history prompt |
| `eval_correct_mistake.py` | streaming metrics eval |
| `scripts/prepare_streaming_data.py` | v2 prefix/chunk JSON |
