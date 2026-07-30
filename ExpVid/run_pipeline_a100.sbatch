#!/usr/bin/env bash
#SBATCH -J expvid_pipe
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/ExpVid/logs/pipe_%j.out
#SBATCH -e /scratch/ll5914/Labos/ExpVid/logs/pipe_%j.err

# ExpVid: prepare frames → zeroshot MCQ → SSL caption NTP → re-eval
set -euo pipefail
echo "[INFO] ExpVid pipeline on $(hostname) at $(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD
pip install -q "transformers>=4.49.0" qwen-vl-utils peft accelerate 2>/dev/null || true

ROOT_CODE=/home/ll5914/Labos/ExpVid
DATA=/scratch/ll5914/Labos/ExpVid
OUT=$DATA/outputs
mkdir -p "$OUT" "$DATA/logs"

# Wait until level_1 videos look complete enough (>=3000 mp4)
echo "[INFO] checking videos..."
for i in $(seq 1 60); do
  N=$(find "$DATA/videos/level_1" -name '*.mp4' 2>/dev/null | wc -l)
  echo "  mp4_count=$N (try $i)"
  if [[ "$N" -ge 3000 ]]; then
    break
  fi
  sleep 60
done
N=$(find "$DATA/videos/level_1" -name '*.mp4' 2>/dev/null | wc -l)
if [[ "$N" -lt 2000 ]]; then
  echo "[ERR] not enough videos yet ($N); requeue later" >&2
  exit 1
fi

echo "========== [1/4] prepare image-caption pairs =========="
python "$ROOT_CODE/scripts/prepare_image_caption.py" \
  --root "$DATA" \
  --ssl-max-train 2000 \
  --val-frac 0.1

echo "========== [2/4] ZERO-SHOT MCQ (image only) =========="
python "$ROOT_CODE/eval_zeroshot_mcq.py" \
  --root "$DATA" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks materials,operation,tools,quantity \
  --out "$OUT/eval_zeroshot_image.json"

echo "========== [2b/4] ZERO-SHOT MCQ (image + caption) =========="
python "$ROOT_CODE/eval_zeroshot_mcq.py" \
  --root "$DATA" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --use-caption \
  --tasks materials,operation,tools,quantity \
  --out "$OUT/eval_zeroshot_image_caption.json"

echo "========== [3/4] SSL caption NTP (LoRA) =========="
python "$ROOT_CODE/train_caption_ssl.py" \
  --root "$DATA" \
  --train-jsonl "$DATA/processed/pairs_train.jsonl" \
  --output-dir "$OUT/caption_ssl_lora" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --epochs 1 \
  --lr 1e-4 \
  --grad-accum 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 100 \
  --log-every 10 \
  --num-workers 4

echo "========== [4/4] POST-SSL MCQ =========="
python "$ROOT_CODE/eval_zeroshot_mcq.py" \
  --root "$DATA" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --checkpoint "$OUT/caption_ssl_lora" \
  --tasks materials,operation,tools,quantity \
  --out "$OUT/eval_postssl_image.json"

python "$ROOT_CODE/eval_zeroshot_mcq.py" \
  --root "$DATA" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --checkpoint "$OUT/caption_ssl_lora" \
  --use-caption \
  --tasks materials,operation,tools,quantity \
  --out "$OUT/eval_postssl_image_caption.json"

echo "========== SUMMARY =========="
python - <<'PY'
import json
from pathlib import Path
out = Path("/scratch/ll5914/Labos/ExpVid/outputs")
for name in [
    "eval_zeroshot_image.json",
    "eval_zeroshot_image_caption.json",
    "eval_postssl_image.json",
    "eval_postssl_image_caption.json",
]:
    p = out / name
    if not p.exists():
        print(name, "MISSING")
        continue
    s = json.loads(p.read_text())["summary"]
    print(f"{name}: acc={s['accuracy']:.4f} n={s['n']} by_task={s['by_task']}")
PY

echo "[INFO] Done at $(date)"
