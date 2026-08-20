#!/usr/bin/env bash
#SBATCH -J wai_lora
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/WearableAI/logs/lora_%x_%j.out
#SBATCH -e /scratch/ll5914/Labos/WearableAI/logs/lora_%x_%j.err

set -euo pipefail

# Usage:
#   sbatch run_finetune_qwen.sbatch
#   sbatch --export=ALL,CONFIG=/home/ll5914/Labos/WearableAI/configs/train_lora_7b.yaml run_finetune_qwen.sbatch

CONFIG="${CONFIG:-/home/ll5914/Labos/WearableAI/configs/train_lora_3b.yaml}"

echo "[INFO] LoRA finetune config=$CONFIG host=$(hostname) date=$(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export HF_TOKEN="$(tr -d '\n' < /home/ll5914/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

# One-time if missing:
# pip install peft accelerate

mkdir -p /scratch/ll5914/Labos/WearableAI/logs
cd /home/ll5914/Labos/WearableAI
python train_lora.py --config "$CONFIG"

echo "[INFO] Done at $(date)"
