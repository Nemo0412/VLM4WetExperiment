#!/usr/bin/env bash
#SBATCH -J fb_qwen_h1
#SBATCH -c 16
#SBATCH -t 01:50:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --signal=B:USR1@90
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/train_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/train_%j.err

set -euo pipefail

ROOT=/home/ll5914/Labos/FineBioQwenStream
DATA=/scratch/ll5914/Labos/FineBioQwenStream/data/proto_prefix_v2
OUT=/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2
CACHE="$DATA/frame_cache"
LOGDIR=/scratch/ll5914/Labos/FineBioQwenStream/logs
SCRIPT="$ROOT/run_train_h100.sbatch"
mkdir -p "$OUT" "$LOGDIR" "$CACHE"

RESUBMIT=1
REQUEUED=0
resubmit_self() {
  if [[ "$REQUEUED" -eq 1 ]]; then return; fi
  if [[ "${RESUBMIT}" -eq 1 && ! -f "$OUT/TRAINING_DONE" ]]; then
    REQUEUED=1
    echo "[INFO] requeue $SCRIPT at $(date) (no TRAINING_DONE)"
    sbatch "$SCRIPT"
  fi
}
trap 'echo "[INFO] caught USR1/TERM at $(date)"; resubmit_self' USR1 TERM

echo "[INFO] FineBioQwenStream SFT H100 (kill-safe) on $(hostname) at $(date) job=${SLURM_JOB_ID:-local}"
nvidia-smi -L

if [[ -f "$OUT/TRAINING_DONE" ]]; then
  echo "[INFO] TRAINING_DONE present — nothing to do"
  RESUBMIT=0
  exit 0
fi

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

pip install -q "transformers>=4.49.0" qwen-vl-utils peft accelerate 2>/dev/null || true

if [ ! -f "$DATA/train.jsonl" ]; then
  python "$ROOT/scripts/prepare_prefix_sft.py" \
    --out-dir "$DATA" --frames-per-proto 4 --halt-frames 5
fi

N_HAVE=$(find "$CACHE" -name '*.npy' 2>/dev/null | wc -l)
if [[ "$N_HAVE" -lt 800 ]]; then
  echo "[INFO] building frame_cache have=${N_HAVE}"
  python "$ROOT/scripts/prepare_frame_cache.py" \
    --data-dir "$DATA" --cache-dir "$CACHE" --workers 16
fi

LOCAL="${SLURM_TMPDIR:-/tmp}/fb_qwen_${SLURM_JOB_ID}"
mkdir -p "$LOCAL/frame_cache"
rsync -a "$DATA/train.jsonl" "$LOCAL/"
rsync -a --info=stats2 "$CACHE/" "$LOCAL/frame_cache/"

(
  while true; do
    date
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
    sleep 30
  done
) > "$LOGDIR/gpu_util_${SLURM_JOB_ID}.log" 2>&1 &
UTIL_PID=$!
trap 'kill $UTIL_PID 2>/dev/null || true; resubmit_self' EXIT

python "$ROOT/train_sft.py" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --train-jsonl "$LOCAL/train.jsonl" \
  --video-root "$DATA" \
  --frame-cache-dir "$LOCAL/frame_cache" \
  --output-dir "$OUT" \
  --max-frames 8 \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --lambda-halt 0.5 \
  --lambda-type 1.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-every 100 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --log-every 5 \
  --resume auto

kill $UTIL_PID 2>/dev/null || true

if [[ -f "$OUT/TRAINING_DONE" ]]; then
  RESUBMIT=0
  echo "[INFO] Done at $(date)"
else
  echo "[INFO] segment ended without TRAINING_DONE — will requeue"
fi
