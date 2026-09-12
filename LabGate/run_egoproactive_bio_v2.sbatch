#!/bin/bash
#SBATCH -J bio_v2
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_v2_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_v2_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/gz2522/VLM4WetExperiment-reference/LabGate
python eval_egoproactive_bio_v2.py \
  --baseline-run /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_16997938 \
  --out-dir /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_v2_${SLURM_JOB_ID}
