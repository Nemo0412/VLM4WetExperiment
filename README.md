# VLM4WetExperiment

Vision–language models for **wet-lab protocol understanding**: compliance / streaming monitoring on [FineBio](https://arxiv.org/abs/2402.00293), plus image–caption SSL on [ExpVid](https://huggingface.co/datasets/OpenGVLab/ExpVid).

Repo: [Nemo0412/VLM4WetExperiment](https://github.com/Nemo0412/VLM4WetExperiment)

---

## Progress (2026-07)

### A. FineBioQwenStream — protocol stream monitor (Qwen2.5-VL-7B)

**Task.** Given intended protocol plan \(P_1{\to}P_2{\to}\cdots\) + a video prefix, decide `CONTINUE` or `HALT`. On `HALT`, report `error_type` ∈ {`missing_protocol`, `wrong_execution`}. Reason text is generated but **not** scored.

**Setup.** LoRA (r=16) on `Qwen/Qwen2.5-VL-7B-Instruct`; vision frozen; `max_frames=8`; NYU HPC A100 with frame cache + USR1 auto-resubmit (kill-safe under ~2h low-util policy).

| Stage | Result |
|-------|--------|
| Zeroshot (FPS ablation) | Decision acc **~5.5%**; et@HALT **0%** (model almost always `CONTINUE`) |
| LoRA SFT v2 (ckpt-1836) | Fixed-index eval: val/test decision **91.3% / 92.1%**, et@HALT **76.7% / 77.7%** |
| Natural streaming rollout | No curated `frame_indices`; FPS on good segs + first *k* bad frames. Test ≤5 detect **99.0%** (1 miss); val **96.9%**. CONTINUE false-HALT still high (~21–32%) |

**Known issue.** Prefix construction concatenates protocol clips **across subjects** (unrealistic for continuous wearables). Next design (v3, not implemented): same-subject sessions, subject-level split, natural sampling, history compression.

**Code:** [`FineBioQwenStream/`](FineBioQwenStream/) — see that folder’s README for train/eval commands.

**Example miss** (`test_r0_neg_wrong_after_6_got3`): P1–P6 correct then wrong P3 instead of P7; model said `CONTINUE` for *k*=1…5. Frames under `FineBioQwenStream/miss_case_frames/`.

### B. ExpVid — image + caption SSL (in progress)

**Task.** Mid-frame image + ASR caption pairs from ExpVid level-1; zero-shot MCQ (image / image+caption); LoRA caption NTP SSL (≤2000 pairs); re-eval.

**Code:** [`ExpVid/`](ExpVid/) — `sbatch ExpVid/run_pipeline_a100.sbatch`. Data lives on scratch (not in git).

### C. Legacy LLaVA-NeXT-Video FineBio pipeline

Earlier track: fine-tune **LLaVA-NeXT-Video-7B** on FineBio for scene / compliance with native multi-frame input (default 32 frames). Scripts remain at repo root (`train_finebio_video.py`, `scripts/prepare_finebio_video.py`, …). Details below.

---

## Repository layout

```
FineBioQwenStream/          # Qwen2.5-VL protocol streaming (current focus)
ExpVid/                     # ExpVid image–caption SSL + MCQ eval
scripts/                    # FineBio download / LLaVA data prep
train_finebio_video.py      # LLaVA-NeXT-Video LoRA + aux heads
train_finebio_ssl.py        # legacy LLaVA-1.5 trainer
infer_protocol_compliance.py
eval_finebio_mistakes.py
run_finebio_*.sbatch
```

Large videos, HF caches, and checkpoints stay on scratch (e.g. `/scratch/$USER/Labos/…`) and are **not** committed.

---

## FineBioQwenStream (quickstart)

```bash
# Build prefix SFT data (videos/annotations on scratch)
python FineBioQwenStream/scripts/prepare_prefix_sft.py \
  --out-dir /scratch/$USER/Labos/FineBioQwenStream/data/proto_prefix_v2 \
  --frames-per-proto 4

# Optional: decode frames to cache (GPU util / kill protection)
python FineBioQwenStream/scripts/prepare_frame_cache.py ...

# Zeroshot FPS ablation, then SFT (A100)
sbatch FineBioQwenStream/run_zeroshot_then_train_a100.sbatch

# Natural streaming eval after SFT
sbatch FineBioQwenStream/run_eval_natural_stream_a100.sbatch
```

---

## Legacy: LLaVA-NeXT-Video on FineBio

Fine-tune **LLaVA-NeXT-Video-7B** for protocol / scene understanding and compliance. Videos are **native multi-frame** (default **32** uniform frames). No frame-grid stitching.

### Requirements

- Python 3.10+, PyTorch, [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (editable install)
- `playwright` + Chromium (Box download), `opencv-python`, `peft`, `transformers`, `decord`, `av`
- 1× A100 (or similar) for training (~1–3 h LoRA, batch=1, 32 frames)

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install -e LLaVA-NeXT
pip install playwright opencv-python peft transformers decord av
playwright install chromium
```

### 1. Download FineBio

```bash
export FINEBIO_BOX_PASSWORD='your_box_password'
export TMPDIR=/path/to/large/tmp

python scripts/download_finebio_box.py \
  --out-dir data/FineBio \
  --password "$FINEBIO_BOX_PASSWORD"
```

Recommended zips: `annotations.zip`, `finebio_videos_fpv_all_w640.zip`, `finebio_mistake_trials.zip`.

### 2. Prepare train / val

```bash
python scripts/prepare_finebio_video.py \
  --videos-dir data/FineBio/videos_w640 \
  --ann-dir data/FineBio/action_annotations \
  --mistake-videos-dir data/FineBio/mistake_videos \
  --out-dir data/finebio_video \
  --num-frames 32 \
  --synth-per-trial 2 \
  --val-frac 0.1
```

### 3. Train

**Model:** `lmms-lab/LLaVA-NeXT-Video-7B-DPO`

\[
L_{\text{total}} = L_{\text{lm}} + \lambda_{\text{comp}} L_{\text{comp}} + 0.3\, L_{\text{proto}}
\]

```bash
python train_finebio_video.py \
  --model-path lmms-lab/LLaVA-NeXT-Video-7B-DPO \
  --train-json data/finebio_video/train.json \
  --video-folder data/finebio_video \
  --output-dir outputs/finebio_video_comp_proto \
  --conv-version vicuna_v1 \
  --num-frames 32 \
  --epochs 3 \
  --lambda-comp 1.0 \
  --lambda-proto 0.3

# Slurm
sbatch run_finebio_video_train_a100.sbatch
```

### 4. Inference

```bash
python infer_protocol_compliance.py \
  --model-path lmms-lab/LLaVA-NeXT-Video-7B-DPO \
  --video path/to/video.mp4 \
  --protocol data/FineBio/protocol_03_dna_extraction.txt \
  --num-frames 32 \
  --output outputs/answer.txt
```

---

## Citation

If you use FineBio or ExpVid, cite the original papers/datasets. This repo is an experimental VLM fine-tuning / evaluation pipeline.
