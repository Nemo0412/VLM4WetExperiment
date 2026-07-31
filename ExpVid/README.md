# ExpVid + Qwen2.5-VL-7B image–caption experiments

Data on scratch (not in git): `/scratch/$USER/Labos/ExpVid`

## Intended task vs current eval

**Intended (next):** given an image from an experiment clip, ask the model what is happening and **generate the caption** (ASR-style narration). Train with **next-token prediction** on `(image, asr_caption)` pairs; evaluate caption quality (e.g. BLEU/CIDEr/ROUGE), not multiple choice.

**Current (proxy, done):** same pairs for LoRA caption NTP, but **evaluation is ExpVid level-1 MCQ** (materials / operation / tools / quantity). The model sees a mid-frame image ± caption text and answers A/B/C/D. This is a temporary probe of whether caption SSL helps visual QA; it is **not** the final caption-generation benchmark.

## Results (job `15066998`, 2026-07-31)

Model: `Qwen/Qwen2.5-VL-7B-Instruct`. Eval: **4035** level-1 MCQs. SSL: LoRA caption NTP on **2000** train pairs (1 epoch).

| Setting | Accuracy |
|---------|----------|
| Zeroshot · image only | **41.98%** |
| Zeroshot · image + caption | **89.34%** |
| Post-SSL · image only | **48.65%** |
| Post-SSL · image + caption | **91.45%** |

Per-task (zeroshot image only → post-SSL image only):

| Task | Zeroshot image | Post-SSL image |
|------|----------------|----------------|
| materials | 34.0% | 42.0% |
| operation | 60.6% | 65.5% |
| quantity | 47.4% | 51.6% |
| tools | 32.2% | 40.3% |

Notes:
- **Image + caption** is still MCQ, but the ASR narration is prepended to the prompt (often leaks answer words) → high accuracy.
- Caption SSL helps **image-only** MCQ (~+6.7 pts); small gain when caption is already in the prompt.
- Artifacts: `/scratch/$USER/Labos/ExpVid/outputs/eval_*.json`, LoRA under `outputs/caption_ssl_lora/`.

## Pipeline (current)

1. Download ExpVid level-1 clips + annotations (mid-frame → image, `asr_caption` → caption).
2. Zero-shot MCQ: image only / image + caption.
3. SSL: LoRA NTP of captions given images.
4. Re-eval the same MCQs with the LoRA adapter.

```bash
# download level_1 (resume-safe)
python scripts/download_level1.py --out-dir /scratch/$USER/Labos/ExpVid

sbatch run_pipeline_a100.sbatch
```

## Layout

| Path | Role |
|--|--|
| `scripts/prepare_image_caption.py` | mid-frame + caption pairs |
| `scripts/download_level1.py` | resume HF level_1 download |
| `eval_zeroshot_mcq.py` | MCQ eval (base / +LoRA) — proxy |
| `train_caption_ssl.py` | LoRA caption NTP |
| `run_pipeline_a100.sbatch` | prepare → zeroshot → SSL → re-eval |
