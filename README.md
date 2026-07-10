# VLM4WetExperiment

Fine-tune **LLaVA-NeXT-Video-7B** on the [FineBio](https://arxiv.org/abs/2402.00293) wet-lab video dataset for **protocol / scene understanding** and **compliance detection**.

Videos are fed as **native multi-frame input** (default **32 uniformly sampled frames**). No frame-grid stitching.

## Requirements

- Python 3.10+, PyTorch, [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) (editable install)
- `playwright` + Chromium (for Box download)
- `opencv-python`, `peft`, `transformers`, `decord`, `av`
- 1× A100 (or similar) for training (~1–3 h with LoRA, batch=1, 32 frames)

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install -e LLaVA-NeXT
pip install playwright opencv-python peft transformers decord av
playwright install chromium
```

## 1. Download FineBio

Official Box link (password from FineBio release page):

```bash
export FINEBIO_BOX_PASSWORD='your_box_password'
export TMPDIR=/path/to/large/tmp   # avoid small /tmp on HPC

python scripts/download_finebio_box.py \
  --out-dir data/FineBio \
  --password "$FINEBIO_BOX_PASSWORD"
```

List files without downloading:

```bash
python scripts/download_finebio_box.py --list-only
```

**Recommended zips:** `annotations.zip`, `finebio_videos_fpv_all_w640.zip` (~6.6 GB, 226 FPV videos), `finebio_mistake_trials.zip`.

Extract videos (Box zips are password-protected; use Python if `unzip -P` fails on your cluster):

```bash
cd data/FineBio
python - <<'PY'
import zipfile, os
pw = b"YOUR_PASSWORD"
for z, sub in [
    ("finebio_videos_fpv_all_w640.zip", "videos_w640"),
    ("annotations.zip", "."),
]:
    with zipfile.ZipFile(z) as zf:
        zf.extractall(sub, pwd=pw) if any(i.flag_bits & 1 for i in zf.infolist()) else zf.extractall(sub)
PY
```

Then unpack nested annotation zip:

```bash
cd data/FineBio
unzip -q annotations/finebio_action_annotations.zip -d action_annotations
```

## 2. Dataset organization

FineBio raw data is converted into **LLaVA instruction JSON** + **video symlinks**. Frames are sampled at train/inference time.

### Raw FineBio inputs

| Source | Content |
|--------|---------|
| `P{xx}_{proto}_{trial}.mp4` | FPV video; middle field `proto` ∈ {01…07} is protocol ID |
| `P{xx}_{proto}_{trial}.txt` | Step annotations: `start_sec,end_sec,task,...` |
| `finebio_mistake_trials.zip` | 3 major mistake videos + 8 minor (paper Table 13) |

### One trial → multiple training samples

For each of **226 trials**, `prepare_finebio_video.py` builds:

```
trial P06_03_01
├── videos/P06_03_01.mp4     # symlink to source mp4
└── JSON entries:
    ├── P06_03_01_intact_scene    # scene QA (correct trials only)
    ├── P06_03_01_intact_comp     # compliance QA (FOLLOWED / NOT FOLLOWED)
    └── P06_03_01_synth{k}_comp   # optional corrupted samples (frame_indices override)
```

**Sample fields** (`train.json` / `val.json`):

```json
{
  "id": "P06_03_01_intact_comp",
  "video": "videos/P06_03_01.mp4",
  "protocol_id": 2,
  "integrity": 1,
  "conversations": [
    {"from": "human", "value": "<image>\n... Did the experimenter follow this protocol?"},
    {"from": "gpt",   "value": "FOLLOWED. The observed steps match ..."}
  ]
}
```

Synthetic corruption samples add `"frame_indices": [12, 45, ...]` so dropped/shuffled steps change which moments are shown.

| Field | Meaning |
|-------|---------|
| `protocol_id` | 0–6 mapped from FineBio protocol 01–07 |
| `integrity` | 1 = intact video, 0 = corrupted / mistake |
| `frame_indices` | optional; overrides uniform 32-frame sampling |

**Split:** shuffle all samples → **90% train / 10% val** (`--val-frac 0.1`, seed=0).

### Frame sampling (default: 32)

LLaVA-NeXT-Video processes each frame separately with spatial pooling (stride 2 → 12×12 tokens/frame). Long context scaling is applied automatically when 32 frames exceed 4096 tokens.

| Model | Default frames | Sampling |
|-------|----------------|----------|
| **LLaVA-NeXT-Video-7B** (this repo) | 32 | Uniform; optional step-aware `frame_indices` for synth |
| Legacy LLaVA-1.5 grid pipeline | 16 | See `scripts/prepare_finebio_llava.py` |

## 3. Prepare train / val split

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

Outputs:

```
data/finebio_video/
  videos/                   # symlinks to mp4 files
  train.json / val.json     # LLaVA conversation format
  protocol_reference.json
  meta.json
```

## 4. Train

**Model:** `lmms-lab/LLaVA-NeXT-Video-7B-DPO`

**Loss:**

\[
L_{\text{total}} = L_{\text{lm}} + \lambda_{\text{comp}} L_{\text{comp}} + 0.3\, L_{\text{proto}}
\]

| Term | Meaning |
|------|---------|
| \(L_{\text{lm}}\) | Standard autoregressive CE on answer tokens |
| \(L_{\text{comp}}\) | 2-class CE: `NOT_FOLLOWED=0`, `FOLLOWED=1` |
| \(L_{\text{proto}}\) | 7-way protocol classification |

Default: \(\lambda_{\text{comp}} = 1.0\). Baseline (LM only): `--lambda-comp 0 --lambda-proto 0`.

**Local / interactive:**

```bash
export HF_HOME=/path/to/huggingface
pip install -e /path/to/LLaVA-NeXT

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
```

**Slurm (A100):**

```bash
sbatch run_finebio_video_train_a100.sbatch
# Baseline only:
MODE=baseline sbatch run_finebio_video_train_a100.sbatch
```

Checkpoints: LoRA adapter + `non_lora_trainables.bin` (projector) + `aux_heads.bin`.

## 5. Inference & evaluation

Protocol check on a single video:

```bash
pip install -e /path/to/LLaVA-NeXT

python infer_protocol_compliance.py \
  --model-path lmms-lab/LLaVA-NeXT-Video-7B-DPO \
  --video path/to/video.mp4 \
  --protocol data/FineBio/protocol_03_dna_extraction.txt \
  --num-frames 32 \
  --output outputs/answer.txt
```

Compare baseline vs fine-tuned on FineBio mistake trials (legacy LLaVA-1.5 eval):

```bash
sbatch run_finebio_eval.sbatch
```

## Project layout

```
scripts/
  download_finebio_box.py      # Box → local/scratch download
  prepare_finebio_video.py     # videos + annotations → LLaVA-NeXT JSON
  prepare_finebio_llava.py     # legacy LLaVA-1.5 grid pipeline
train_finebio_video.py         # LLaVA-NeXT-Video LoRA + aux heads
train_finebio_ssl.py           # legacy LLaVA-1.5 trainer
infer_protocol_compliance.py   # single-video inference (LLaVA-NeXT-Video)
eval_finebio_mistakes.py       # baseline vs fine-tuned comparison (legacy)
run_finebio_video_train_a100.sbatch
run_finebio_ssl_train*.sbatch  # legacy
```

## Citation

If you use FineBio, cite the original dataset paper. This repo provides a VLM fine-tuning pipeline on top of LLaVA-NeXT-Video.
