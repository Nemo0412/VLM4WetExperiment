#!/bin/bash
#SBATCH -J bio_native
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_native_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_native_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/gz2522/VLM4WetExperiment-reference/LabGate
DATA=/home/gz2522/bio-dataset/EgoProactive-Bio
BASE=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate
EVAL="${BIO_NATIVE_EVAL:-$BASE/data/egoproactive_bio_native_20260905.jsonl}"
OUT="$BASE/outputs/bio_native_${SLURM_JOB_ID}"
JUDGER=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
EXPERT=/scratch/ll5914/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da
mkdir -p "$BASE/data" "$BASE/logs"
cd "$ROOT"
if [[ ! -f "$EVAL" ]]; then
  python prepare_egoproactive_bio_labgate.py --dataset "$DATA" --out "$EVAL"
fi
python eval_egoproactive_bio_native.py --eval-jsonl "$EVAL" --out-dir "$OUT" \
  --judger "$JUDGER" --expert "$EXPERT" --always-expert
