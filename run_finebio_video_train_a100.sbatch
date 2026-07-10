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
sleep 15
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

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
