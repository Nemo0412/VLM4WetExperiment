#!/usr/bin/env bash
#SBATCH -J fb_qwen_nat
#SBATCH -c 8
#SBATCH -t 01:50:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/eval_nat_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/eval_nat_%j.err

# Natural streaming eval: video + protocol only (FPS sample, no curated indices)
set -euo pipefail
echo "[INFO] natural stream eval on $(hostname) at $(date)"
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
CKPT=/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2/checkpoint-1836
OUT=/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2
mkdir -p "$OUT" /scratch/ll5914/Labos/FineBioQwenStream/logs

for SPLIT in val test; do
  echo "========== NATURAL STREAM $SPLIT =========="
  python "$ROOT/eval_stream_natural.py" \
    --checkpoint "$CKPT" \
    --data-jsonl "$DATA/${SPLIT}.jsonl" \
    --video-root "$DATA" \
    --sample-fps 1.0 \
    --max-frames 8 \
    --max-frames-per-seg 8 \
    --max-pixels $((128 * 28 * 28)) \
    --min-pixels $((4 * 28 * 28)) \
    --out "$OUT/eval_natural_stream_${SPLIT}_ckpt1836.json"
done

echo "[INFO] Done at $(date)"
python - <<'PY'
import json
from pathlib import Path
out = Path("/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2")
for split in ("val", "test"):
    p = out / f"eval_natural_stream_{split}_ckpt1836.json"
    s = json.loads(p.read_text())["summary"]
    print(f"{split}: detect@≤5={s['detect_rate']:.3f} mean_lat={s['latency_mean']} "
          f"et@first={s['et_acc_at_first_halt']} false_halt={s['continue_false_halt_rate']:.3f}")
PY
