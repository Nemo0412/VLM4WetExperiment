#!/usr/bin/env bash
#SBATCH -J finebio_dl
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH --mem=8G
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH -o /scratch/ll5914/logs/finebio_dl_%j.out
#SBATCH -e /scratch/ll5914/logs/finebio_dl_%j.err

set -euo pipefail

echo "[INFO] FineBio Box download started on $(hostname) at $(date)"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

mkdir -p /scratch/ll5914/logs /scratch/ll5914/Labos/Llava/data/FineBio

# Full dataset (~157GB). To download only what you need for LLaVA first, pass --only, e.g.:
#   --only annotations finebio_videos_fpv_train.zip finebio_videos_fpv_valid.zip
python /home/ll5914/Labos/Llava/scripts/download_finebio_box.py \
  --out-dir /scratch/ll5914/Labos/Llava/data/FineBio

echo "[INFO] Done at $(date)"
