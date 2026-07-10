#!/usr/bin/env bash
#SBATCH -J finebio_eval
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/logs/finebio_eval_%j.out
#SBATCH -e /scratch/ll5914/logs/finebio_eval_%j.err

set -euo pipefail
echo "[INFO] FineBio eval on $(hostname) at $(date)"
export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd /home/ll5914/Labos/Llava
python eval_finebio_mistakes.py \
  --num-frames 16 \
  --skip-zero-shot \
  --out-dir /scratch/ll5914/Labos/Llava/outputs/finebio_eval_compare

echo "[INFO] Done at $(date)"
