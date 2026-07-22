#!/usr/bin/env bash
#SBATCH -J fb_qwen_sft
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/train_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/train_%j.err

set -euo pipefail
echo "[INFO] FineBioQwenStream SFT on $(hostname) at $(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

pip install -q "transformers>=4.49.0" qwen-vl-utils peft accelerate 2>/dev/null || true

ROOT=/home/ll5914/Labos/FineBioQwenStream
DATA=/scratch/ll5914/Labos/FineBioQwenStream/data/proto_prefix_v1
OUT=/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2
mkdir -p "$OUT" /scratch/ll5914/Labos/FineBioQwenStream/logs

if [ ! -f "$DATA/train.jsonl" ]; then
  python "$ROOT/scripts/prepare_prefix_sft.py" --out-dir "$DATA"
fi

LOCAL="${SLURM_TMPDIR:-/tmp}/fb_qwen_${SLURM_JOB_ID}"
mkdir -p "$LOCAL"
rsync -a "$DATA/train.jsonl" "$LOCAL/"

# OOM fixes vs v1: max_frames 48→16, max_pixels capped, LoRA r 64→16,
# last-hidden hook (no full hidden_states), grad-accum 16→8
python "$ROOT/train_sft.py" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --train-jsonl "$LOCAL/train.jsonl" \
  --video-root "$DATA" \
  --output-dir "$OUT" \
  --max-frames 8 \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lambda-halt 0.5 \
  --lambda-type 1.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 200 \
  --num-workers 2 \
  --log-every 5

echo "[INFO] Done at $(date)"
