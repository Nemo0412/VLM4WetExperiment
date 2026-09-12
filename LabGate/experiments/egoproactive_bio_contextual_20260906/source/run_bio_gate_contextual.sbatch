#!/bin/bash
#SBATCH -J bio_gate_ctx
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_gate_ctx_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_gate_ctx_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/home/gz2522/VLM4WetExperiment-reference/LabGate
BASE=/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate
JUDGER=/scratch/gz2522/gz2522/hf_cache/qwen25vl-led/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3
cd "$ROOT"
python eval_egoproactive_bio_gate_contextual.py \
  --eval-jsonl "$BASE/data/bio_contextual_official_20260906/gate_contextual.jsonl" \
  --out-dir "$BASE/outputs/bio_gate_contextual_${SLURM_JOB_ID}" --judger "$JUDGER"
