#!/usr/bin/env bash
#SBATCH -J fb_qwen_both
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/both_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/both_%j.err

# (1) zeroshot FPS ablation  (2) LoRA SFT
set -euo pipefail
echo "[INFO] FineBioQwenStream FPS-ablation+SFT on $(hostname) at $(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

pip install -q "transformers>=4.49.0" qwen-vl-utils peft accelerate 2>/dev/null || true

ROOT=/home/ll5914/Labos/FineBioQwenStream
DATA=/scratch/ll5914/Labos/FineBioQwenStream/data/proto_prefix_v2
OUT_ROOT=/scratch/ll5914/Labos/FineBioQwenStream/outputs
OUT_SFT="$OUT_ROOT/qwen_proto_sft_v2"
OUT_ABL="$OUT_ROOT/fps_ablation_zeroshot"
mkdir -p "$OUT_ROOT" "$OUT_SFT" "$OUT_ABL" /scratch/ll5914/Labos/FineBioQwenStream/logs

if [ ! -f "$DATA/train.jsonl" ]; then
  python "$ROOT/scripts/prepare_prefix_sft.py" \
    --out-dir "$DATA" --frames-per-proto 4 --halt-frames 5
fi

echo "========== [1/2] ZERO-SHOT FPS ablation =========="
# Grid: stored indices + 0.25/0.5/1/2/4 FPS
# max_frames=8 keeps compute comparable; per-seg cap=8 so one long proto can fill budget
python "$ROOT/eval_prefix.py" \
  --zero-shot \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --data-jsonl "$DATA/val.jsonl" \
  --video-root "$DATA" \
  --max-frames 8 \
  --max-frames-per-seg 8 \
  --fps-grid "stored,0.25,0.5,1,2,4" \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --out "$OUT_ABL"

# also write a convenience copy of best/common baseline path
if [ -f "$OUT_ABL/eval_stored.json" ]; then
  cp -f "$OUT_ABL/eval_stored.json" "$OUT_ROOT/eval_val_zeroshot.json"
fi

echo "========== [2/2] LoRA SFT (max_frames=8) =========="
LOCAL="${SLURM_TMPDIR:-/tmp}/fb_qwen_${SLURM_JOB_ID}"
mkdir -p "$LOCAL"
rsync -a "$DATA/train.jsonl" "$LOCAL/"

python "$ROOT/train_sft.py" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --train-jsonl "$LOCAL/train.jsonl" \
  --video-root "$DATA" \
  --output-dir "$OUT_SFT" \
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
  --save-every 200 \
  --num-workers 2 \
  --log-every 5

echo "[INFO] All done at $(date)"
