#!/usr/bin/env bash
#SBATCH -J fb_qwen_fps
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/fps_abl_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/fps_abl_%j.err

# Standalone zeroshot FPS ablation (no training).
set -euo pipefail
echo "[INFO] zeroshot FPS ablation on $(hostname) at $(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

pip install -q "transformers>=4.49.0" qwen-vl-utils peft accelerate 2>/dev/null || true

ROOT=/home/ll5914/Labos/FineBioQwenStream
DATA=/scratch/ll5914/Labos/FineBioQwenStream/data/proto_prefix_v2
OUT=/scratch/ll5914/Labos/FineBioQwenStream/outputs/fps_ablation_zeroshot
mkdir -p "$OUT" /scratch/ll5914/Labos/FineBioQwenStream/logs

python "$ROOT/eval_prefix.py" \
  --zero-shot \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --data-jsonl "$DATA/val.jsonl" \
  --video-root "$DATA" \
  --max-frames 8 \
  --max-frames-per-seg 8 \
  --fps-grid "stored,0.25,0.5,1,2,4" \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --out "$OUT"

echo "[INFO] Done at $(date)"
