#!/usr/bin/env bash
#SBATCH -J finebio_llava_32f
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/finebio_llava_32f_%j.out
#SBATCH -e /scratch/ll5914/logs/finebio_llava_32f_%j.err

set -euo pipefail

echo "[INFO] FineBio protocol test (32 frames) started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME=/scratch/ll5914/huggingface
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

ROOT=/home/ll5914/Labos/Llava
TEST_DIR=/scratch/ll5914/Labos/Llava/outputs/finebio_test
PROTOCOL=/scratch/ll5914/Labos/Llava/data/FineBio/protocol_03_dna_extraction.txt
NUM_FRAMES=32
mkdir -p "$TEST_DIR" /scratch/ll5914/logs

run_case() {
  local name="$1"
  local video="$2"
  local out_dir="$TEST_DIR/${name}_32f"
  mkdir -p "$out_dir"
  echo ""
  echo "========== $name (32 frames) =========="
  echo "video=$video"
  python "$ROOT/infer_protocol_compliance.py" \
    --model-path liuhaotian/llava-v1.5-7b \
    --video "$video" \
    --protocol "$PROTOCOL" \
    --num-frames "$NUM_FRAMES" \
    --temperature 0.0 \
    --max-new-tokens 512 \
    --grid-out "$out_dir/frame_grid.jpg" \
    --output "$out_dir/llava_answer.txt"
}

run_case "P06_03_01_correct" "$TEST_DIR/P06_03_01.mp4"
run_case "P06_03_02_mistake" "$TEST_DIR/P06_03_02.mp4"

echo ""
echo "[INFO] Done at $(date)"
