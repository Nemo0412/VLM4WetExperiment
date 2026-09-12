#!/bin/bash
#SBATCH -J bio_expert_ctx
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_expert_ctx_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_expert_ctx_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/gz2522/VLM4WetExperiment-reference/LabGate
BASE=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate
EXPERT=/scratch/ll5914/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da
cd "$ROOT"
python eval_egoproactive_bio_expert_contextual.py \
  --eval-jsonl "$BASE/data/bio_contextual_official_20260906/expert_contextual.jsonl" \
  --out-dir "$BASE/outputs/bio_expert_contextual_${SLURM_JOB_ID}" --expert "$EXPERT"
