#!/bin/bash
#SBATCH -J bio_n_diag
#SBATCH -c 8
#SBATCH -t 00:15:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_n_diag_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_n_diag_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
ROOT=/home/gz2522/VLM4WetExperiment-reference/LabGate
BASE=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate
JUDGER=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
EVAL="${BIO_NATIVE_EVAL:-$BASE/data/egoproactive_bio_native_20260905.jsonl}"
test -f "$EVAL"
cd "$ROOT"
python diagnose_egoproactive_bio_native.py --eval-jsonl "$EVAL" \
  --out-dir "$BASE/outputs/bio_native_diag_${SLURM_JOB_ID}" --judger "$JUDGER"
