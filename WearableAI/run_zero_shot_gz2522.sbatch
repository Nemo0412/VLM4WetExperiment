#!/usr/bin/env bash
#SBATCH -J wai_zs_3b
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/zero_shot_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI/logs/zero_shot_%j.err

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TAG="${TAG:-zero_shot_3b_official}"
ROOT=/home/gz2522/VLM4WetExperiment-reference/WearableAI
SCRATCH=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/WearableAI
DATA=/scratch/gz2522/datasets/wearable-ai
KIT="$DATA/starter_kit"
OUT="$SCRATCH/outputs/$TAG"

source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HOME=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

mkdir -p "$OUT" "$SCRATCH/logs"
echo "[INFO] model=$MODEL max_samples=${MAX_SAMPLES:-full} host=$(hostname) date=$(date)"
nvidia-smi -L

extra=()
if [[ -n "$MAX_SAMPLES" ]]; then
  extra+=(--max-samples "$MAX_SAMPLES" --no-eval)
fi

cd "$KIT"
python run_evaluation.py \
  --task proactive \
  --model-type qwen \
  --llm-model "$MODEL" \
  --backend hf \
  --video-folder "$DATA/egoproactive/val" \
  --golden "$DATA/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl" \
  --predictions "$OUT/predictions.jsonl" \
  --eval-output "$OUT/results.json" \
  "${extra[@]}"

echo "[INFO] done date=$(date) output=$OUT"