# EgoProactive-Bio 解耦评估：3B Gate 与 Oracle-gated 32B Expert

> 本目录保留 Gate-only 和 Oracle-gated Expert 的原始证据。与端到端结果合并后的统一说明见
> [`../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。
> 本运行未向模型提供数据集 history/current step，现仅作为缺失上下文消融；修正版见
> [`../egoproactive_bio_contextual_20260906/`](../egoproactive_bio_contextual_20260906/README.md)。

## 目的

原生级联的 32B 只被调用 1/25 次，不能用该结果判断大模型本身能否给出正确指导。
本实验将两个问题拆开：

```text
Gate-only（25 条）
视频 + SOP + 空 ASR → 3B → YES/NO + reason → 只评分触发

Oracle-expert（14 条正例）
视频 + SOP + oracle reason → 32B → TYPE + MSG → 只评分类型和指导
```

`oracle reason` 只是一个正确小模型理论上会传出的粗粒度类别：`safety`、
`action_error` 或 `next_step`。32B 没有收到 decision task、参考指导语、当前步骤名称或
语义 rubric。

## 数据集构造与信息边界

[`prepare_egoproactive_bio_decoupled.py`](../../prepare_egoproactive_bio_decoupled.py)
从同一份不可修改的 EgoProactive-Bio 标注生成两个文件：

| 数据集 | 数量 | 模型输入 | 仅评分使用 |
|---|---:|---|---|
| gate_eval | 25 | video、16 个 frame indices、SOP、空 ASR | gt_fire、gt_reason、phase/pair |
| expert_oracle_eval | 14 | video、frame indices、SOP、空 ASR、oracle_reason | gt_type、参考指导、语义 rubric、phase/pair |

- Gate 集包含 14 个 interrupt 和 11 个 silent；不含参考指导文本。
- Expert 集只包含 14 个 interrupt；不存在 silent/是否调用的判断。
- `oracle_reason` 是本实验有意提供的 oracle 输入，不能计入端到端级联成绩。
- 每条语义 rubric 在推理前固定为若干必要概念组。例如 `d03` 必须同时覆盖 UV 和关闭；
  rubric 只对生成文本评分，不进入 prompt。
- 标注 SHA-256：`0486d4c8091c1b406c8e2358bef5ba6f61e5ffabac3bb438a5eabe135f999968`。

## 实验条件

3B Gate 测两个互斥画面条件：

1. `native_reprojection`：原 LabGate `reproject_keep(tau=0.12)`。
2. `full_16_frames`：不筛帧，保留相同窗口内全部 16 帧。

32B Expert 测四个条件：

1. 重投影帧 + 正确 oracle reason。
2. 完整 16 帧 + 正确 oracle reason。
3. 重投影帧 + `reason=unknown`，只表示 gate 已触发。
4. 完整 16 帧 + `reason=unknown`。

第 3、4 组用于判断结果变化来自正确 reason，还是只要强制调用 32B 就能得到。
所有条件继续使用原 `expert_prompt`、同一 BF16 权重和 greedy decoding。

## 3B Gate-only 结果

作业 `17000761`，L40S，`COMPLETED 0:0`，2 分 07 秒，50 次 3B 调用。

| 画面条件 | Accuracy | Precision | Recall | F1 | TP/FP/FN/TN | reason 正确率（14 个正例） |
|---|---:|---:|---:|---:|---:|---:|
| 原重投影 | 44.0% | 0 | 0 | 0 | 0/0/14/11 | 0% |
| 完整 16 帧 | 48.0% | 1.000 | 7.1% | 0.133 | 1/0/13/11 | 0% |

完整 16 帧只触发 `d23`；二分类属于 TP，但模型输出 `reason=action_error`，而该点真实
reason 是 `next_step`，所以 reason 正确率仍是 0。这个结果单独确认了 3B gate 是主要瓶颈，
与 32B 的输出质量无关。

## Oracle-gated 32B 结果

作业 `17000868`，H200 `gh124`，`COMPLETED 0:0`，4 分 49 秒，56 次 32B 调用。

| 画面 / reason | TYPE Accuracy | 完整概念覆盖 | 平均概念召回 | 非空指导 |
|---|---:|---:|---:|---:|
| 原重投影 + oracle reason | 71.4% | 14.3%（2/14） | 26.2% | 100% |
| 完整 16 帧 + oracle reason | 64.3% | 14.3%（2/14） | 26.2% | 100% |
| 原重投影 + reason unknown | 7.1% | 7.1% | 16.7% | 100% |
| 完整 16 帧 + reason unknown | 7.1% | 14.3% | 23.8% | 100% |

正确 reason 将 TYPE Accuracy 从 7.1% 提高到 64.3%–71.4%。这说明 reason 传递对输出
类别影响很大；同时 `safety/action_error/next_step` 与目标 TYPE 本身存在直接映射，因此
这个提升不能被解释为 32B 已看懂了具体视觉动作。真正反映指导内容的概念覆盖仍然很低。

Oracle reason 下逐条非盲语义复核采用严格标准：指导必须给出参考标注要求的当前动作，
不能只选对 TYPE 或提到相关词。

| 画面条件 | correct | partial | incorrect |
|---|---:|---:|---:|
| 原重投影 | 2/14 | 1/14 | 11/14 |
| 完整 16 帧 | 1/14 | 2/14 | 11/14 |

- 两种画面都正确：`d03`，关闭 UV。
- 只有重投影条件正确：`d12`，添加 trypsin。
- `d01` 包含从 incubator 取 flask，但重复了已经完成的手部消毒，记 partial。
- 完整帧的 `d21` 识别到 transfer，却直接指导后续加 medium/label，记 partial。
- 其余典型错误包括把未戴手套误说成 UV 开启、把 PBS rinse 跳到 trypsin、把过度
  trypsin 暴露误说成未戴手套，以及在最后步骤重新讨论 PBS/trypsin。

关键词完整覆盖在完整帧条件把 `d21` 算为通过，而严格审核只算 partial。因此 14.3%
自动指标对该条件仍偏乐观。逐条审核不是盲法领域专家评审，只用于指出自动关键词指标的边界；
完整原文均已留档，可继续由实验人员复核。

## 结论

解耦后可以分别回答两个问题：

1. **3B 不能可靠触发。** 原重投影 recall 为 0，完整 16 帧 recall 为 7.1%，且唯一 TP
   的 reason 仍错误。
2. **假设正确触发并给出正确粗粒度 reason，32B 通常能选到合理 TYPE，但不能可靠生成
   当前步骤的正确指导。** 严格内容正确率只有 1–2/14。
3. **瓶颈不只在 gate。** 直接绕过 gate 后，32B 仍缺少 protocol progress；粗粒度
   `next_step` 没有告诉它刚完成的是哪一步，视频窗口也不足以稳定恢复此前状态。
4. **完整 16 帧没有解决状态问题。** 它对个别样本有帮助，但整体 TYPE 和概念指标没有提高。

下一版如果希望回答“大模型在拥有充分上游信息时能否指导”，应增加第三个独立上界：给 32B
传入由非 GT 状态追踪器生成的 `current_step/completed_step`。如果直接传 GT step，则必须明确
命名为 oracle-step upper bound，不能作为真实系统结果。

## 复现与留档

```bash
cd /home/gz2522/VLM4WetExperiment-reference
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  LabGate/prepare_egoproactive_bio_decoupled.py \
  --dataset /home/gz2522/bio-dataset/EgoProactive-Bio \
  --out-dir /path/to/new-empty-data-dir
sbatch LabGate/run_bio_gate_decoupled.sbatch
sbatch LabGate/run_bio_expert_oracle.sbatch
```

- [数据构造元数据](dataset_metadata.json)、[Gate 数据](gate_eval.jsonl)、
  [Oracle-expert 数据](expert_oracle_eval.jsonl)
- [Gate summary](gate_summary.json)、[50 条 Gate 输出](gate_predictions.jsonl)、
  [Gate run](gate_run.json)
- [Expert summary](expert_summary.json)、[56 条 Expert 输出](expert_predictions.jsonl)、
  [Expert run](expert_run.json)
- [Slurm 作业元数据](job_metadata.json)
- [逐条严格审核](manual_review.csv)
- `source/` 保存本次实际运行的构造、评估和 Slurm 脚本快照。
