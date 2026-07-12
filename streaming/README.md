# FineBioStreaming

Video stream + protocol → **CONTINUE** / **HALT + reason** (stop on HALT).

Backbone: `LLaVA-NeXT-Video-7B-DPO` + LoRA + aux `halt_head` / `step_head`.

## Quickstart

```bash
# prepare chunk data
python scripts/prepare_streaming_data.py

# train (4× A100)
sbatch run_train_a100.sbatch

# stream one video
python infer_stream.py \
  --checkpoint /path/to/ckpt \
  --video /path/to/video.mp4 \
  --protocol-id 3 \
  --halt-threshold 0.45

# correct vs mistake pairs
python eval_correct_mistake.py --checkpoint /path/to/ckpt
```

## Layout

| File | Role |
|------|------|
| `train_streaming.py` | LoRA + aux heads |
| `infer_stream.py` | chunk loop → early stop |
| `eval_correct_mistake.py` | 3-pair correct/mistake eval |
| `scripts/prepare_streaming_data.py` | prefix/chunk JSON |
