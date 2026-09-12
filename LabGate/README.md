# LabGate

Zero-shot wet-lab monitor with a cheap frame gate, a small Judger VLM, and a
large expert VLM:

```text
Audio ──► Audio2Text ──────────────────────┐
Video ──► reprojection ─► Judger (Qwen2.5-VL-3B) ─YES─► Expert (32B)
Protocol ──────────────────────────────────┘                 │
                                                SAFETY / ACTION_ERROR / ASSISTANT
```

The Judger only calls the 32B expert for safety issues, user questions, or
protocol conflicts. `FineBioWhen2See/gate.py` removes frames explained by
viewpoint reprojection.

## Zero-shot accuracy test

`eval_zeroshot.py` is the accuracy/latency script. It evaluates 42 FineBio
cases across `none`, `assistant`, `safety`, and `action_error`, comparing the
3B→32B gate with always running 32B.

On the NYU cluster, run:

```bash
cd LabGate
sbatch run_zeroshot_32b_h200.sbatch
```

The job builds the cases with `prepare_eval.py`, then runs:

```bash
python eval_zeroshot.py \
  --eval-jsonl /scratch/ll5914/Labos/LabGate/data/eval_v2.jsonl \
  --judger Qwen/Qwen2.5-VL-3B-Instruct \
  --expert Qwen/Qwen2.5-VL-32B-Instruct \
  --always-expert \
  --out /scratch/ll5914/Labos/LabGate/outputs/zeroshot_3b32b.json
```

Measured zero-shot result: **76.2%** end-to-end type accuracy, **0.82** gate
F1, and **0.93 s** gated latency versus **1.45 s** always-32B latency.

Run one clip:

```bash
python run_one.py \
  --video /path/to/video.mp4 \
  --protocol-id 3 \
  --asr "What should I do next?" \
  --t0 20 --t1 28
```

FineBio videos, model weights, predictions, and logs are intentionally excluded
from Git.

## EgoProactive-Bio: small VLM decision → large VLM guidance

全部数据构造、模型实际输入/输出、prompt、指标、端到端与解耦结果、逐条指导审核和失败原因，
统一见 **[EgoProactive-Bio 完整实验报告](EGOPROACTIVE_BIO_EVALUATION_REPORT.md)**。
下面只保留快速入口；各 `experiments/` README 是原始运行留档。

当前入口是上下文修正版：`dialog[i]` 按官方 query + 最近 4 turn 结构作为真实 chat history，
`task[i]` 作为相同 `current_step` 传给 3B、32B 和 32B trigger JSON；仅 3B parsed YES
调用 32B，并转发实际预测 reason。运行：

```bash
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  -m unittest discover -s LabGate -p 'test_*.py' -v
sbatch LabGate/run_bio_gate_contextual.sbatch
sbatch --partition=h100_tandon LabGate/run_bio_expert_contextual.sbatch
sbatch --partition=h100_tandon LabGate/run_bio_contextual_cascade.sbatch
```

主条件实测为 TP/FP/FN/TN = 5/4/9/7，Gate Recall 35.7%，F1 0.435；9 次
32B 调用中，5 个真阳性的指导为 4 correct / 1 partial，4 个误触发全部产生错误或
不必要指导。Oracle-triggered 32B 的 TYPE accuracy 为 64.3%，人工指导为
6 correct / 4 partial / 4 wrong。3B 仍会在 raw history 的过度触发与 clean history
的过度 silent 之间摆动。

完整 prompt、标签映射、原始输出、诊断和原因分析见
[上下文实验记录](experiments/egoproactive_bio_contextual_20260906/README.md)和
[上下文 prompts](PROMPTS_BIO_CONTEXTUAL.md)。[仓库组件核查](REPOSITORY_AUDIT.md)说明其他文件夹中
哪些流程可复用，以及为什么没有直接套用相邻项目的 LoRA 或状态机。

前两版保留作为对照：

- [v1](experiments/egoproactive_bio_20260905/README.md)：专用 `$silent$/$interrupt$` prompt 和 gold/rollout 历史。
- [v2](experiments/egoproactive_bio_v2_20260905/README.md)：JSON 理由/观察/步骤状态；实测出现观察复制退化。
