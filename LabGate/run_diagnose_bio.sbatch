#!/bin/bash
#SBATCH -J bio_diag
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/diag_%j.out
#SBATCH -e /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/diag_%j.err
set -euo pipefail
source /scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
cd /home/gz2522/VLM4WetExperiment-reference/LabGate
python diagnose_egoproactive_bio.py \
  --run /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_16997938 \
  --out /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/diag_${SLURM_JOB_ID}
