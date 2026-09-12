#!/usr/bin/env bash
#SBATCH -J wai_lora_3b
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/lora_train_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/lora_train_%j.err

set -euo pipefail

ROOT=/home/gz2522/VLM4WetExperiment-reference/WearableAI
SCRATCH=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI
CONFIG="${CONFIG:-$ROOT/configs/train_lora_3b_gz2522.yaml}"

source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HOME=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led
export WEARABLE_AI_STARTER_KIT=/scratch/gz2522/datasets/wearable-ai/starter_kit
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

mkdir -p "$SCRATCH/logs"
echo "[INFO] config=$CONFIG host=$(hostname) date=$(date)"
nvidia-smi -L
cd "$ROOT"
python train_lora.py --config "$CONFIG"
echo "[INFO] done date=$(date)"