# VLM4WetExperiment

Fine-tune **LLaVA-1.5** on the [FineBio](https://arxiv.org/abs/2402.00293) wet-lab video dataset for **protocol / scene understanding** and **compliance detection**.

## Requirements

- Python 3.10+, PyTorch, [LLaVA](https://github.com/haotian-liu/LLaVA) (editable install)
- `playwright` + Chromium (for Box download)
- `opencv-python`, `peft`, `transformers`
- 1 GPU for training (~20 min with LoRA on A100/H100)

```bash
pip install playwright opencv-python peft transformers
playwright install chromium
# Install LLaVA from source, then use your conda env
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

FineBio raw data is converted into **LLaVA instruction-tuning JSON** + **frame-grid images**. We do **not** feed raw video to LLaVA-1.5 directly.

### Raw FineBio inputs

| Source | Content |
|--------|---------|
| `P{xx}_{proto}_{trial}.mp4` | FPV video; middle field `proto` ∈ {01…07} is protocol ID |
| `P{xx}_{proto}_{trial}.txt` | Step annotations: `start_sec,end_sec,task,...` |
| `finebio_mistake_trials.zip` | 3 major mistake videos + 8 minor (paper Table 13) |

### One trial → multiple training samples

For each of **226 trials**, `prepare_finebio_llava.py` builds:

```
trial P06_03_01
├── {trial}_intact.jpg          # uniform 16-frame grid from full video
├── {trial}_synth0.jpg          # optional: drop-one-step corruption
├── {trial}_synth1.jpg          # optional: shuffle-step corruption
└── JSON entries:
    ├── {trial}_intact_scene    # scene QA (correct trials only)
    └── {trial}_intact_comp     # compliance QA (FOLLOWED / NOT FOLLOWED)
```

**Sample fields** (`train.json` / `val.json`):

```json
{
  "id": "P06_03_01_intact_comp",
  "image": "images/P06_03_01_intact.jpg",
  "protocol_id": 2,
  "integrity": 1,
  "conversations": [
    {"from": "human", "value": "<image>\n... Did the experimenter follow this protocol?"},
    {"from": "gpt",   "value": "FOLLOWED. The observed steps match ..."}
  ]
}
```

| Field | Meaning |
|-------|---------|
| `protocol_id` | 0–6 mapped from FineBio protocol 01–07 |
| `integrity` | 1 = intact video, 0 = corrupted / mistake |
| `_scene` | Protocol + step description (answer **generated from annotations**) |
| `_comp` | Compliance verdict (FOLLOWED=1 / NOT_FOLLOWED=0 for aux head) |

**Labels:** protocol ID comes from the filename; QA text is **template-generated** from step CSVs (FineBio has no official protocol text or dialogue labels).

**Synthetic errors:** for correct trials, we programmatically build corrupted grids (missing / shuffled steps) to augment the 11 real mistake videos.

**Split:** shuffle all samples → **90% train / 10% val** (`--val-frac 0.1`, seed=0). Typical counts: ~989 train, ~109 val, ~1098 total.

### Frame sampling (current default: 16)

LLaVA-1.5 is **image-only**: N frames are sampled uniformly in time, resized, and tiled into **one square grid** (e.g. 16 → 4×4, each cell ~336 px).

| Frames | Grid | Issue |
|--------|------|-------|
| 16 | 4×4 | Current default; ~1 frame per protocol step |
| 32 | ~6×6 | Smaller cells, more temporal coverage |
| 256 | 16×16 | **Not usable as one grid** — each cell ≈ 21 px, unreadable; image would be ~5376×5376 px |

**256 frames** requires a different design, e.g.:

- **Multi-chunk:** 256 frames → 16 grids × 16 frames, multi-turn QA or aggregate compliance head
- **Annotation-aware sampling:** sample at step boundaries instead of uniform (better than blind 256)
- **Video LLM:** LLaVA-NeXT-Video / Video-LLaVA (native multi-frame input)

This repo currently implements **uniform sampling + single grid** only (`--num-frames`, default **16**).

## 3. Prepare train / val split

Build frame grids + LLaVA instruction JSON (default **90% train / 10% val**):

```bash
python scripts/prepare_finebio_llava.py \
  --videos-dir data/FineBio/videos_w640/finebio_videos_w640 \
  --ann-dir data/FineBio/action_annotations/finebio_action_annotations \
  --mistake-videos-dir data/FineBio/mistake_videos \
  --out-dir data/finebio_llava \
  --num-frames 16 \
  --synth-per-trial 3 \
  --val-frac 0.1
```

Outputs:

```
data/finebio_llava/
  images/                   # one grid JPG per trial variant
  train.json / val.json     # LLaVA conversation format
  protocol_reference.json   # inferred step lists per protocol 01–07
```

## 4. Train

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

python train_finebio_ssl.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --train-json data/finebio_llava/train.json \
  --image-folder data/finebio_llava \
  --output-dir outputs/finebio_comp_proto \
  --epochs 3 \
  --lambda-comp 1.0 \
  --lambda-proto 0.3
```

**Slurm (A100 example):**

```bash
sbatch run_finebio_ssl_train_a100.sbatch
# Baseline only:
MODE=baseline sbatch run_finebio_ssl_train_a100.sbatch
```

Checkpoints: LoRA adapter + `non_lora_trainables.bin` (projector) + `aux_heads.bin`.

## 5. Inference & evaluation

Zero-shot or fine-tuned protocol check on a single video:

```bash
python infer_protocol_compliance.py \
  --model-path liuhaotian/llava-v1.5-7b \
  --video path/to/video.mp4 \
  --protocol data/FineBio/protocol_03_dna_extraction.txt \
  --num-frames 16 \
  --output outputs/answer.txt
```

Compare baseline vs fine-tuned on FineBio mistake trials:

```bash
sbatch run_finebio_eval.sbatch
# Report: outputs/finebio_eval_compare/report.md
```

## Project layout

```
scripts/
  download_finebio_box.py    # Box → local/scratch download
  prepare_finebio_llava.py   # videos + annotations → LLaVA JSON
train_finebio_ssl.py         # LoRA fine-tuning + aux heads
infer_protocol_compliance.py # single-video inference
eval_finebio_mistakes.py     # baseline vs fine-tuned comparison
run_finebio_ssl_train*.sbatch
```

## Citation

If you use FineBio, cite the original dataset paper. This repo provides a VLM fine-tuning pipeline on top of LLaVA.
