#!/usr/bin/env bash
#SBATCH -J wai_w2p32
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/ll5914/Labos/WearableAI/logs/w2p_%x_%j.out
#SBATCH -e /scratch/ll5914/Labos/WearableAI/logs/w2p_%x_%j.err

set -euo pipefail
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}"
exec bash /home/ll5914/Labos/WearableAI/run_when2prompt_qwen.sbatch
