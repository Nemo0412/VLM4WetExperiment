#!/usr/bin/env bash
#SBATCH -J labgate_bio
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_%j.err

set -euo pipefail
ROOT="${LABGATE_ROOT:-/home/gz2522/VLM4WetExperiment-reference/LabGate}"
DATA="${BIO_DATASET:-/home/gz2522/bio-dataset/EgoProactive-Bio}"
OUT="${BIO_OUT:-/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_${SLURM_JOB_ID}}"
JUDGER="${JUDGER:-/scratch/gz2522/gz2522/hf_cache/qwen25vl-led/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
EXPERT="${EXPERT:-/scratch/ll5914/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da}"
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HOME=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
nvidia-smi
python eval_egoproactive_bio.py --dataset "$DATA" --out-dir "$OUT" \
  --judger "$JUDGER" --expert "$EXPERT" --history-modes gold rollout
