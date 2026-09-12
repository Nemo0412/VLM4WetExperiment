#!/bin/bash
#SBATCH -J bio_ctx_cascade
#SBATCH -c 8
#SBATCH -t 00:35:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_ctx_cascade_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_ctx_cascade_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/home/gz2522/VLM4WetExperiment-reference/LabGate
BASE=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate
JUDGER=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
EXPERT=/scratch/ll5914/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da
cd "$ROOT"
python eval_egoproactive_bio_contextual_cascade.py \
  --gate-jsonl "$BASE/data/bio_contextual_official_20260906/gate_contextual.jsonl" \
  --expert-jsonl "$BASE/data/bio_contextual_official_20260906/expert_contextual.jsonl" \
  --out-dir "$BASE/outputs/bio_contextual_cascade_${SLURM_JOB_ID}" \
  --judger "$JUDGER" --expert "$EXPERT"
