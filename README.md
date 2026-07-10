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

## 2. Prepare train / val split

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
  images/          # 16-frame grids per trial
  train.json       # ~989 samples
  val.json         # ~109 samples
  protocol_reference.json
```

Each sample includes `protocol_id` (0–6) and `integrity` (1=intact, 0=corrupted) for auxiliary losses.

## 3. Train

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

## 4. Inference & evaluation

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
