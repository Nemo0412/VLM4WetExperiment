#!/usr/bin/env bash
#SBATCH -J wai_w2p
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH -o /scratch/ll5914/Labos/WearableAI/logs/w2p_%x_%j.out
#SBATCH -e /scratch/ll5914/Labos/WearableAI/logs/w2p_%x_%j.err

set -euo pipefail

# Usage: sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct run_when2prompt_qwen.sbatch
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # empty = full 700 sessions
TAG="${TAG:-$(echo "$LLM_MODEL" | tr '/' '_' | tr '.' 'p')}"

echo "[INFO] Wearable-AI EgoProactive vanilla zero-shot when2prompt"
echo "[INFO] model=$LLM_MODEL host=$(hostname) date=$(date)"
nvidia-smi -L

export HF_HOME=/scratch/ll5914/huggingface
export HF_TOKEN="$(tr -d '\n' < /home/ll5914/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

KIT=/scratch/ll5914/datasets/wearable-ai/starter_kit
DATA=/scratch/ll5914/datasets/wearable-ai
OUT=/scratch/ll5914/Labos/WearableAI/outputs/${TAG}
mkdir -p "$OUT" /scratch/ll5914/Labos/WearableAI/logs

nvid=$(ls "$DATA/egoproactive/val"/*.mp4 2>/dev/null | wc -l)
echo "[INFO] videos present: $nvid / 700"
if [[ "$nvid" -lt 700 ]]; then
  echo "[ERROR] EgoProactive videos not fully downloaded" >&2
  exit 1
fi

extra=()
if [[ -n "$MAX_SAMPLES" ]]; then
  extra+=(--max-samples "$MAX_SAMPLES")
fi

cd "$KIT"
python run_evaluation.py \
  --task proactive \
  --model-type qwen \
  --llm-model "$LLM_MODEL" \
  --backend hf \
  --video-folder "$DATA/egoproactive/val" \
  --golden "$DATA/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl" \
  --predictions "$OUT/predictions.jsonl" \
  --eval-output "$OUT/results.json" \
  "${extra[@]}"

echo "[INFO] Done at $(date)"
OUT_DIR="$OUT" python - <<'PY'
import json, os
out = os.environ["OUT_DIR"]
path = os.path.join(out, "results_summary.json")
if not os.path.exists(path):
    path = os.path.join(out, "results.json")
r = json.load(open(path))
ov = r.get("overall", r)
tp, tn = ov.get("tp", 0), ov.get("tn", 0)
sup = ov.get("support", 0) or 0
ov["accuracy"] = round((tp + tn) / sup, 4) if sup else 0.0
print("=== when2prompt (interrupt vs silent) ===")
for k in ["accuracy", "macro_f1", "gmean_f1", "interrupt_f1", "silent_f1",
          "interrupt_precision", "interrupt_recall", "silent_precision",
          "silent_recall", "tp", "fp", "tn", "fn", "support"]:
    if k in ov:
        print(f"  {k}: {ov[k]}")
json.dump(ov, open(os.path.join(out, "when2prompt_metrics.json"), "w"), indent=2)
print("wrote", os.path.join(out, "when2prompt_metrics.json"))
PY
