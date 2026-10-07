#!/usr/bin/env bash
#SBATCH -J fb_intr72
#SBATCH -c 16
#SBATCH -t 16:00:00
#SBATCH --mem=384G
#SBATCH --gres=gpu:h200:2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioInterrupt/logs/annotate_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioInterrupt/logs/annotate_%j.err

set -euo pipefail

echo "[INFO] FineBio interrupt annotation Qwen2.5-VL-72B on 2xH200 $(hostname) $(date)"
mkdir -p /scratch/ll5914/Labos/FineBioInterrupt/logs \
         /scratch/ll5914/Labos/FineBioInterrupt/annotations \
         /scratch/ll5914/tmp

export HF_HOME=/scratch/ll5914/huggingface
export HF_TOKEN="$(tr -d '\n' < /home/ll5914/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/ll5914/tmp

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

nvidia-smi -L

ROOT=/home/ll5914/Labos/FineBioInterrupt
cd "$ROOT"

# Keep both allocated GPUs visibly busy while weights download.
python -u keepalive_gpu.py &
KP=$!
cleanup() { kill "$KP" 2>/dev/null || true; wait "$KP" 2>/dev/null || true; }
trap cleanup EXIT

python - << 'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-VL-72B-Instruct")
print("[INFO] model ready", path, flush=True)
PY

kill "$KP" 2>/dev/null || true
wait "$KP" 2>/dev/null || true
trap - EXIT

python -u annotate_interrupts.py \
  --ann-dir /scratch/ll5914/Labos/Llava/data/FineBio/action_annotations \
  --video-dir /scratch/ll5914/Labos/Llava/data/FineBio/videos_w640 \
  --out-dir /scratch/ll5914/Labos/FineBioInterrupt/annotations \
  --model Qwen/Qwen2.5-VL-72B-Instruct

echo "[INFO] Done at $(date)"
