#!/usr/bin/env bash
#SBATCH -J labgate32
#SBATCH -c 8
#SBATCH -t 03:50:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH -o /scratch/ll5914/Labos/LabGate/logs/zs32_%j.out
#SBATCH -e /scratch/ll5914/Labos/LabGate/logs/zs32_%j.err

set -euo pipefail
echo "[INFO] LabGate zeroshot 3B→32B on $(hostname) at $(date)"
nvidia-smi -L

export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD
pip install -q "transformers>=4.49.0" qwen-vl-utils accelerate 2>/dev/null || true

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA=/scratch/ll5914/Labos/LabGate/data/eval_v2.jsonl
OUT=/scratch/ll5914/Labos/LabGate/outputs
mkdir -p "$OUT" /scratch/ll5914/Labos/LabGate/logs

python "$ROOT/prepare_eval.py" --out "$DATA"
python "$ROOT/eval_zeroshot.py" \
  --eval-jsonl "$DATA" \
  --judger Qwen/Qwen2.5-VL-3B-Instruct \
  --expert Qwen/Qwen2.5-VL-32B-Instruct \
  --always-expert \
  --out "$OUT/zeroshot_3b32b.json"
