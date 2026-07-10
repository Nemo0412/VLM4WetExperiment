#!/usr/bin/env bash
#SBATCH -J lsv_llava7b
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/%x_%j.out
#SBATCH -e /scratch/ll5914/logs/%x_%j.err

set -euo pipefail

echo "[INFO] Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME=/scratch/ll5914/huggingface
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

ROOT=/home/ll5914/Labos/Llava
VIDEO="$ROOT/data/LSV/DJI/DJI-Video/DJI_20250723141140_0009_D.MP4"
PROTOCOL="$ROOT/data/LSV/DJI/DJI-Protocol/Demo_Cas9_delivery.txt"
OUT_DIR="$ROOT/outputs/DJI-014"
mkdir -p "$OUT_DIR" /scratch/ll5914/logs

echo "[INFO] Sample: LSV DJI-014 (HF viewer row=13)"
echo "[INFO] Operation: CRISPR Delivery"
echo "[INFO] Protocol file: Demo_Cas9_delivery.txt"
echo "[INFO] Issue annotation: none (expected FOLLOWED)"

python "$ROOT/infer_protocol_compliance.py" \
  --model-path liuhaotian/llava-v1.5-7b \
  --video "$VIDEO" \
  --protocol "$PROTOCOL" \
  --num-frames 8 \
  --temperature 0.0 \
  --max-new-tokens 512 \
  --grid-out "$OUT_DIR/frame_grid.jpg" \
  --output "$OUT_DIR/llava_answer.txt"

echo "[INFO] Done at $(date)"
