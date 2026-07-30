#!/usr/bin/env bash
#SBATCH -J fb_qwen_eval
#SBATCH -c 8
#SBATCH -t 01:50:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/FineBioQwenStream/logs/eval_sft_%j.out
#SBATCH -e /scratch/ll5914/Labos/FineBioQwenStream/logs/eval_sft_%j.err

# Eval LoRA SFT checkpoint on val + test (stored indices, same as training).
set -euo pipefail
echo "[INFO] FineBioQwenStream SFT eval on $(hostname) at $(date)"
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

if [[ ! -f "$CKPT/adapter_config.json" ]]; then
  echo "[ERR] missing checkpoint: $CKPT" >&2
  exit 1
fi

echo "========== VAL (stored indices) =========="
python "$ROOT/eval_prefix.py" \
  --checkpoint "$CKPT" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --data-jsonl "$DATA/val.jsonl" \
  --video-root "$DATA" \
  --max-frames 8 \
  --max-frames-per-seg 8 \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --out "$OUT/eval_val_ckpt1836.json"

echo "========== TEST (stored indices) =========="
python "$ROOT/eval_prefix.py" \
  --checkpoint "$CKPT" \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --data-jsonl "$DATA/test.jsonl" \
  --video-root "$DATA" \
  --max-frames 8 \
  --max-frames-per-seg 8 \
  --max-pixels $((128 * 28 * 28)) \
  --min-pixels $((4 * 28 * 28)) \
  --out "$OUT/eval_test_ckpt1836.json"

echo "[INFO] Done at $(date)"
python - <<'PY'
import json
from pathlib import Path
out = Path("/scratch/ll5914/Labos/FineBioQwenStream/outputs/qwen_proto_sft_v2")
for name in ("eval_val_ckpt1836.json", "eval_test_ckpt1836.json"):
    p = out / name
    if not p.exists():
        print(f"missing {p}")
        continue
    s = json.loads(p.read_text())["summary"]
    print(f"{name}: dec={s['decision_accuracy']:.4f} et@halt={s['error_type_accuracy_on_halt']:.4f} "
          f"n={s['n']} conf={s['confusion_gt_pred']}")
PY
