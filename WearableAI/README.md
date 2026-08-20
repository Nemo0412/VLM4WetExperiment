# Wearable-AI EgoProactive (when2prompt)

Vanilla zero-shot and LoRA finetune experiments on **EgoProactive** from [facebook/wearable-ai](https://huggingface.co/datasets/facebook/wearable-ai), using the **official starter_kit proactive protocol**.

Task: at each ~8s video chunk, predict `$interrupt$<guidance>` or `$silent$`.

## Zero-shot results (official protocol)

Evaluated on the full **700 videos / 9935 chunk decisions** with:

| Setting | Value |
|---------|-------|
| Starter kit | `/scratch/ll5914/datasets/wearable-ai/starter_kit/run_evaluation.py --task proactive` |
| Golden | `wearable_ai_2026_egoproactive_val_700.jsonl` |
| Video | cumulative frames 0..j, 16 frames/interval, max 32 |
| History | gold `dialog[j]` (max 4 turns after query) |
| Decoding | greedy, `max_new_tokens=512` |
| Metrics | Macro F1, G-mean F1, accuracy = (tp+tn)/support |

Gold label balance: ~53% interrupt / ~47% silent.

| Model | Accuracy | Macro F1 | G-mean F1 | Int F1 | Sil F1 | Int recall | Sil recall | Pred interrupt |
|-------|----------|----------|-----------|--------|--------|------------|------------|------------------|
| Qwen2.5-VL-3B | 0.481 | 0.389 | 0.308 | 0.626 | 0.152 | 0.806 | 0.101 | **~85%** |
| Qwen2.5-VL-7B | 0.461 | 0.459 | 0.458 | 0.489 | 0.429 | 0.479 | 0.439 | **~52%** |
| Qwen2.5-VL-32B | 0.459 | 0.370 | 0.283 | 0.607 | 0.132 | 0.776 | 0.089 | **~84%** |

Outputs:

```
/scratch/ll5914/Labos/WearableAI/outputs/qwen25vl_3b/
/scratch/ll5914/Labos/WearableAI/outputs/qwen25vl_7b/
/scratch/ll5914/Labos/WearableAI/outputs/Qwen_Qwen2p5-VL-32B-Instruct/
```

**Takeaways**

- **7B** is the only size that stays near the gold interrupt rate (~52% predicted vs ~53% gold); 3B/32B zero-shot **over-trigger** (~85% interrupt).
- High interrupt recall + low silent recall on 3B/32B → models behave like always-on coaches, not timing-aware assistants.
- Macro F1 alone can look OK on 3B (0.39) because interrupt F1 is high; G-mean F1 (0.31) exposes silent-class collapse.

### Re-run zero-shot

```bash
# 3B / 7B on A100
sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct run_when2prompt_qwen.sbatch
sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct run_when2prompt_qwen.sbatch

# 32B on H200
sbatch run_when2prompt_qwen32b.sbatch

# Debug subset
sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct,MAX_SAMPLES=10 run_when2prompt_qwen.sbatch
```

---

## Data layout

```
/scratch/ll5914/datasets/wearable-ai/
├── starter_kit/                          # official HF eval scripts
└── egoproactive/
    ├── wearable_ai_2026_egoproactive_val_700.jsonl
    └── val/*.mp4                         # 700 videos (~23GB)
```

Download (gated — accept terms on HF first):

```bash
export HF_TOKEN=$(cat /home/ll5914/.hf_token)
python download_egoproactive.py
```

---

## LoRA finetune (Wearable-AI protocol)

Public EgoProactive only ships a **val** split. For finetune we **split 700 videos by video_id** (85% train / 15% val, seed=42) so no video leaks between train and val. This is an internal dev protocol; for benchmark comparison, zero-shot numbers above use the full 700.

Training matches inference:

- Same `SYSTEM_PROMPT`, cumulative frames, gold dialog history
- Target = gold `$interrupt$ …` or `$silent$`
- LoRA on `q_proj, k_proj, v_proj, o_proj` (rank 32)

### Pipeline

```bash
# 1) Prepare SFT JSONL (~8454 train / ~1481 val decisions)
sbatch run_prepare_sft.sbatch

# 2) LoRA train (default 3B config)
sbatch run_finetune_qwen.sbatch
sbatch --export=ALL,CONFIG=/home/ll5914/Labos/WearableAI/configs/train_lora_7b.yaml run_finetune_qwen.sbatch

# 3) Eval LoRA on full 700 (official metrics via starter_kit)
sbatch --export=ALL,\
BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct,\
ADAPTER=/scratch/ll5914/Labos/WearableAI/outputs/lora_3b/best,\
TAG=lora_3b run_eval_finetuned.sbatch
```

### Local (non-Slurm) commands

```bash
conda activate /scratch/ll5914/conda_envs/SVD
cd /home/ll5914/Labos/WearableAI

# deps (once)
pip install peft accelerate

python prepare_sft_data.py \
  --golden /scratch/ll5914/datasets/wearable-ai/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \
  --out-dir /scratch/ll5914/Labos/WearableAI/data/sft

python train_lora.py --config configs/train_lora_3b.yaml

python eval_lora.py \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/best \
  --golden /scratch/ll5914/datasets/wearable-ai/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \
  --video-folder /scratch/ll5914/datasets/wearable-ai/egoproactive/val \
  --predictions /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/predictions.jsonl

cd /scratch/ll5914/datasets/wearable-ai/starter_kit
python run_evaluation.py --task proactive --eval-only \
  --golden ../egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl \
  --predictions /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/predictions.jsonl \
  --eval-output /scratch/ll5914/Labos/WearableAI/outputs/lora_3b/results.json
```

### Finetune outputs

```
/scratch/ll5914/Labos/WearableAI/
├── data/sft/{train,val}.jsonl
└── outputs/lora_{3b,7b}/
    ├── best/              # lowest val-loss LoRA adapter
    ├── last/
    ├── metrics.jsonl
    ├── predictions.jsonl
    ├── results.json
    └── when2prompt_metrics.json
```

### Config knobs

Edit `configs/train_lora_3b.yaml` / `train_lora_7b.yaml`:

| Key | Default | Notes |
|-----|---------|-------|
| `data.frames_per_interval` | 16 | official proactive default |
| `data.max_frames` | 32 | cumulative cap |
| `data.max_history_turns` | 4 | gold dialog history |
| `training.num_epochs` | 2 | |
| `training.gradient_accumulation_steps` | 16 | effective batch 16 |
| `model.image_max_pixels` | 112896 | caps vision tokens / avoids OOM |

32B LoRA is not pre-configured (needs H200 + lower resolution or QLoRA).

---

## Project files

| File | Purpose |
|------|---------|
| `run_when2prompt_qwen.sbatch` | Zero-shot eval (3B/7B) |
| `run_when2prompt_qwen32b.sbatch` | Zero-shot eval (32B, H200) |
| `download_egoproactive.py` | Download gated videos to scratch |
| `proactive_protocol.py` | Shared message/frame helpers (starter_kit aligned) |
| `prepare_sft_data.py` | JSONL → SFT train/val |
| `train_lora.py` | LoRA SFT |
| `eval_lora.py` | LoRA inference → predictions.jsonl |
| `run_prepare_sft.sbatch` | Slurm: prepare data |
| `run_finetune_qwen.sbatch` | Slurm: train |
| `run_eval_finetuned.sbatch` | Slurm: eval + official metrics |

---

## Environment

- Conda: `/scratch/ll5914/conda_envs/SVD`
- HF cache: `/scratch/ll5914/huggingface`
- HF token: `/home/ll5914/.hf_token`
- Account/partition: `torch_pr_674_tandon_advanced` / `a100_tandon` (32B: `h200_tandon`)

Logs: `/scratch/ll5914/Labos/WearableAI/logs/`
