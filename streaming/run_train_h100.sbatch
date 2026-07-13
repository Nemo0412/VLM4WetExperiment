#!/usr/bin/env bash
#SBATCH -J fb_stream_v2
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH --mem=384G
#SBATCH --gres=gpu:4
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioStreaming/logs/train_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioStreaming/logs/train_%j.err

set -euo pipefail

echo "[INFO] FineBioStreaming v2 4-GPU H100 train on $(hostname) at $(date)"
sleep 5
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

LLAVA_NEXT=/scratch/ll5914/Labos/LLaVA-NeXT
pip install -q -e "$LLAVA_NEXT" 2>/dev/null || pip install -e "$LLAVA_NEXT"

ROOT=/home/ll5914/Labos/FineBioStreaming
DATA=/scratch/ll5914/Labos/FineBioStreaming/data/streaming_v2
OUT=/scratch/ll5914/Labos/FineBioStreaming/outputs/streaming_vlm_v2
MODEL="${MODEL:-lmms-lab/LLaVA-NeXT-Video-7B-DPO}"
mkdir -p "$OUT" /scratch/ll5914/Labos/FineBioStreaming/logs

if [ ! -f "$DATA/train.json" ]; then
  python "$ROOT/scripts/prepare_streaming_data.py" --out-dir "$DATA"
fi

(
  while true; do
    date
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
    sleep 30
  done
) > /scratch/ll5914/Labos/FineBioStreaming/logs/gpu_util_${SLURM_JOB_ID}.log 2>&1 &
UTIL_PID=$!
trap 'kill $UTIL_PID 2>/dev/null || true' EXIT

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  "$ROOT/train_streaming.py" \
  --model-path "$MODEL" \
  --train-json "$DATA/train.json" \
  --video-folder "$DATA" \
  --output-dir "$OUT" \
  --conv-version vicuna_v1 \
  --num-frames 8 \
  --pool-stride 2 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 2 \
  --lr 1e-4 \
  --lambda-halt 1.0 \
  --lambda-step 0.3 \
  --attn-impl sdpa \
  --num-workers 4 \
  --save-every 100 \
  --log-every 10

echo "[INFO] Done at $(date)"
