#!/usr/bin/env bash
#SBATCH -J fb_eval
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioStreaming/logs/eval_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioStreaming/logs/eval_%j.err

set -euo pipefail

echo "[INFO] FineBioStreaming eval on $(hostname) at $(date)"
sleep 5
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

LLAVA_NEXT=/scratch/ll5914/Labos/LLaVA-NeXT
pip install -q -e "$LLAVA_NEXT" 2>/dev/null || pip install -e "$LLAVA_NEXT"

ROOT=/home/ll5914/Labos/FineBioStreaming
cd "$ROOT"

python eval_correct_mistake.py \
  --checkpoint /scratch/ll5914/Labos/FineBioStreaming/outputs/streaming_vlm_v1 \
  --out-dir /scratch/ll5914/Labos/FineBioStreaming/outputs/eval_correct_mistake \
  --halt-threshold 0.45 \
  --num-frames 8 \
  --window-sec 20

echo "[INFO] Done at $(date)"
