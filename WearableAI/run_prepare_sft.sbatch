#!/usr/bin/env bash
#SBATCH -J wai_sft_prep
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/WearableAI/logs/sft_prep_%j.out
#SBATCH -e /scratch/ll5914/Labos/WearableAI/logs/sft_prep_%j.err

set -euo pipefail

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

DATA=/scratch/ll5914/datasets/wearable-ai/egoproactive
OUT=/scratch/ll5914/Labos/WearableAI/data/sft
mkdir -p "$OUT" /scratch/ll5914/Labos/WearableAI/logs

cd /home/ll5914/Labos/WearableAI
python prepare_sft_data.py \
  --golden "$DATA/wearable_ai_2026_egoproactive_val_700.jsonl" \
  --out-dir "$OUT"

echo "[INFO] Done at $(date)"
