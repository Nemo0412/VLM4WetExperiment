#!/usr/bin/env bash
#SBATCH -J wai_eval_3b
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/lora_eval_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/lora_eval_%j.err

set -euo pipefail

ROOT=/home/gz2522/VLM4WetExperiment-reference/WearableAI
SCRATCH=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI
DATA=/scratch/gz2522/datasets/wearable-ai
KIT="$DATA/starter_kit"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
ADAPTER="${ADAPTER:-$SCRATCH/outputs/lora_3b/best}"
TAG="${TAG:-lora_3b_official_full700}"
OUT="$SCRATCH/outputs/$TAG"

source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HOME=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led
export WEARABLE_AI_STARTER_KIT="$KIT"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

mkdir -p "$OUT" "$SCRATCH/logs"
echo "[INFO] base=$BASE_MODEL adapter=$ADAPTER host=$(hostname) date=$(date)"
nvidia-smi -L

cd "$ROOT"
python eval_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --golden "$DATA/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl" \
  --video-folder "$DATA/egoproactive/val" \
  --predictions "$OUT/predictions.jsonl"

cd "$KIT"
python run_evaluation.py \
  --task proactive \
  --eval-only \
  --golden "$DATA/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl" \
  --predictions "$OUT/predictions.jsonl" \
  --eval-output "$OUT/results.json"

echo "[INFO] done date=$(date) output=$OUT"