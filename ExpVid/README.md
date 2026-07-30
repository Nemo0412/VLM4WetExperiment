# ExpVid + Qwen2.5-VL-7B image–caption experiments

Data on scratch (not in git): `/scratch/$USER/Labos/ExpVid`

## Pipeline

1. Download **ExpVid** level-1 clips + annotations (video mid-frame → image, `asr_caption` → caption).
2. **Zero-shot** Qwen2.5-VL-7B on level-1 MCQ (materials / operation / tools / quantity):
   - image only
   - image + caption in the prompt
3. **SSL**: LoRA next-token prediction of captions given images (subset, default 2000 pairs).
4. Re-evaluate the same MCQs with the LoRA adapter.

## Run

```bash
# download (if needed)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('OpenGVLab/ExpVid', repo_type='dataset',
  local_dir='/scratch/$USER/Labos/ExpVid',
  allow_patterns=['videos/level_1/**','annotations/**','README.md'])
PY

sbatch run_pipeline_a100.sbatch
```

Outputs: `/scratch/$USER/Labos/ExpVid/outputs/`.

## Layout

| Path | Role |
|--|--|
| `scripts/prepare_image_caption.py` | mid-frame + caption pairs |
| `eval_zeroshot_mcq.py` | MCQ eval (base / +LoRA) |
| `train_caption_ssl.py` | LoRA caption NTP |
| `run_pipeline_a100.sbatch` | prepare → zeroshot → SSL → re-eval |
