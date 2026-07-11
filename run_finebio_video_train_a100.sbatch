#!/usr/bin/env bash
#SBATCH -J finebio_vid
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/logs/finebio_video_%j.out
#SBATCH -e /scratch/ll5914/logs/finebio_video_%j.err

set -euo pipefail

echo "[INFO] FineBio LLaVA-NeXT-Video training (v2 stable) on $(hostname) at $(date)"
sleep 10
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Do NOT override CUDA_VISIBLE_DEVICES — Slurm binds the allocated GPU for this job.
echo "[INFO] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

LLAVA_NEXT=/scratch/ll5914/Labos/LLaVA-NeXT
pip install -q -e "$LLAVA_NEXT" 2>/dev/null || pip install -e "$LLAVA_NEXT"

ROOT=/home/ll5914/Labos/Llava
DATA=/scratch/ll5914/Labos/Llava/data/finebio_video
MODE="${MODE:-ssl}"
EPOCHS="${EPOCHS:-3}"
MODEL="${MODEL:-lmms-lab/LLaVA-NeXT-Video-7B-DPO}"

if [ ! -f "$DATA/train.json" ]; then
  echo "[INFO] Preparing finebio_video dataset..."
  python "$ROOT/scripts/prepare_finebio_video.py" \
    --videos-dir /scratch/ll5914/Labos/Llava/data/FineBio/videos_w640 \
    --ann-dir /scratch/ll5914/Labos/Llava/data/FineBio/action_annotations \
    --mistake-videos-dir /scratch/ll5914/Labos/Llava/data/FineBio/mistake_videos \
    --out-dir "$DATA" \
    --num-frames 16 \
    --synth-per-trial 2
fi

if [ "$MODE" = "baseline" ]; then
  LCOMP=0.0; LPROTO=0.0
  OUT="${OUT:-/scratch/ll5914/Labos/Llava/outputs/finebio_video_baseline_v2}"
else
  LCOMP=1.0; LPROTO=0.3
  OUT="${OUT:-/scratch/ll5914/Labos/Llava/outputs/finebio_video_comp_proto_v2}"
fi

echo "[INFO] MODE=$MODE model=$MODEL out=$OUT epochs=$EPOCHS frames=16"

python3 - <<'PY'
import os, sys, torch
print("[cuda-preflight] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[cuda-preflight] torch.cuda.is_available =", torch.cuda.is_available())
print("[cuda-preflight] device_count =", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("CUDA preflight failed: no GPU visible to PyTorch")
x = torch.zeros(1, device="cuda:0")
print("[cuda-preflight] ok, tensor device =", x.device)
PY

python "$ROOT/train_finebio_video.py" \
  --model-path "$MODEL" \
  --train-json "$DATA/train.json" \
  --video-folder "$DATA" \
  --output-dir "$OUT" \
  --conv-version vicuna_v1 \
  --num-frames 16 \
  --pool-stride 2 \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --proj-lr 2e-5 \
  --lora-r 64 --lora-alpha 128 \
  --lambda-comp "$LCOMP" \
  --lambda-proto "$LPROTO" \
  --max-len 4096 \
  --max-grad-norm 0.5 \
  --attn-impl sdpa \
  --log-every 10

echo "[INFO] Done at $(date)"
