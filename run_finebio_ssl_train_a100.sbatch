#!/usr/bin/env bash
#SBATCH -J finebio_ssl
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/logs/finebio_ssl_%j.out
#SBATCH -e /scratch/ll5914/logs/finebio_ssl_%j.err

set -euo pipefail

echo "[INFO] FineBio LM+comp+proto training on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export HF_HOME=/scratch/ll5914/huggingface
export TMPDIR=/scratch/ll5914/tmp
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

ROOT=/home/ll5914/Labos/Llava
DATA=/scratch/ll5914/Labos/Llava/data/finebio_llava
MODE="${MODE:-ssl}"
EPOCHS="${EPOCHS:-3}"

if [ "$MODE" = "baseline" ]; then
  LCOMP=0.0; LPROTO=0.0
  OUT="${OUT:-/scratch/ll5914/Labos/Llava/outputs/finebio_lora_baseline_v2}"
else
  LCOMP=1.0; LPROTO=0.3
  OUT="${OUT:-/scratch/ll5914/Labos/Llava/outputs/finebio_comp_proto}"
fi

echo "[INFO] MODE=$MODE L_total=lm + ${LCOMP}*comp + ${LPROTO}*proto out=$OUT epochs=$EPOCHS"

python "$ROOT/train_finebio_ssl.py" \
  --model-path liuhaotian/llava-v1.5-7b \
  --train-json "$DATA/train.json" \
  --image-folder "$DATA" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --proj-lr 2e-5 \
  --lora-r 64 --lora-alpha 128 \
  --lambda-comp "$LCOMP" \
  --lambda-proto "$LPROTO" \
  --log-every 10

echo "[INFO] Done at $(date)"
