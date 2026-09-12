# EgoProactive-Bio · LabGate 3B → 32B 实验记录

> 这是 v1 历史基线，不是当前推荐流程。统一结论和最终输入输出定义见
> [`../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。

后续的输入审查与小模型对照实验见 [原因分析](analysis.md)。

## 目标与代码版本

按“小 VLM 判断 silent/interrupt；仅在 yes/interrupt 后调用大 VLM 输出指导语句”
评估本地 `bio-dataset/EgoProactive-Bio`。这是零样本、标注决策点上的两级模型评估，没有训练或调参。

- 日期：2026-09-05。
- 仓库：`Nemo0412/VLM4WetExperiment`，本地 `/home/gz2522/VLM4WetExperiment-reference`。
- 执行 `git fetch origin`、`git merge --ff-only origin/main`，从 `a758448` 更新到 `5755752`。
- 保留原有 `WearableAI/proactive_protocol.py` 的本地修改及四个未跟踪配置/脚本。
- 本次新增的代码在上述提交之上；每次运行在 `run.json` 记录关键源文件 SHA-256。

## 数据与输入边界

- 数据版本：`0.3.0-draft`，一个 726 秒、960×1280、30 fps、无音频的合成视频。
- 标注：25 个决策，14 interrupt / 11 silent；6 组错误→修复。
- `scripts/validate_dataset.py` 实际执行通过，包括视频 SHA-256、时间线、嵌套历史和标签一致性。
- 只在 25 个给定决策时间评价，不是逐帧或固定步长连续监测，因此没有检测延迟/提前量指标。
- 模型输入包括画面、任务 `Cell passaging`、静态 SOP、帧时间戳和既往对话。
- 静态 SOP 整理自数据自带的 `source_docs/cell_passaging_protocol_and_action_v2.docx`，
  包含该文档中的规范流程和通用注意事项；没有当前步骤编号、视频错误片段编号、观察时间或答案。
- 当前 `task`、`answer`、`phase`、`pair_id`、错误解释均只用于评分/审核，不传入模型。
- `gold`：使用数据集提供的决策前完整真实对话，是有真实历史条件的评估。
- `rollout`：只从用户任务开始，累积本模型实际输出的指导语句，不输入真实历史。
  视频仍是固定录制，不会对模型建议作出反应，因此也不是交互式闭环实验。

## 实现

```text
因果历史采样 + 当前窗口采样
    → LabGate 重投影筛帧（强制保留最新帧）
    → Qwen2.5-VL-3B：$silent$ / $interrupt$
        ├─ silent/no：不调用大模型，不输出指导语句
        ├─ invalid：记为格式错误，不调用大模型，评分计错
        └─ interrupt/yes：Qwen2.5-VL-32B → 1–2 句英文指导
```

- 复用 `LabGate/models.py` 中的 Qwen 包装器、`pipeline.reproject_keep` 和 `FineBioWhen2See/gate.py`。
- 使用新 `egoproactive_bio.py` 提示词：除了危险/动作错误，也允许在步骤完成后给出下一步提示。
  原 LabGate 仅因危险、冲突或提问触发，与本数据的正常步骤主动提示标签不完全匹配。
- 原 `pipeline.sample_window` 取窗口中点附近相邻帧；本入口改用覆盖整个窗口的均匀采样，
  默认 8 张当前窗口帧 + 最多 8 张前缀帧。全部时间严格小于当前决策时间。
- 重投影阈值 `tau=0.12`；额外保留最后一帧，以便观察当前状态。
- 两级模型接收相同的筛后画面和上下文；小模型不生成指导，大模型不再重新决定是否 silent。
- 解析器只识别开头的决策词，接受 `$interrupt$`、`interrupt`、`interupt`、`YES`；
  `$silent$`、`silent`、`NO` 不触发。正文中偶然出现 YES 不会触发。
- 模型 BF16、SDPA、`device_map=auto`、greedy decoding；小模型最多 16 token，大模型最多 128 token。
- 每帧 `max_pixels=128*28*28`、`min_pixels=4*28*28`，沿用 LabGate 包装器。
- 实际时间戳写入提示词；底层视频 processor 的默认时间编码沿用原包装器，非均匀采样的
  精确运动速度/时长理解不保证可靠，特别影响“摇晃过猛”和“孵育过久”的判断。

## 执行与复现

```bash
cd /home/gz2522/VLM4WetExperiment-reference
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  /home/gz2522/bio-dataset/EgoProactive-Bio/scripts/validate_dataset.py
python -m unittest discover -s LabGate -p 'test_egoproactive_bio.py' -v
mkdir -p /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs
sbatch LabGate/run_egoproactive_bio_h200.sbatch
```

- 路由、时间边界、上下文边界、invalid 计分 4 项 CPU 测试通过。
- Slurm 作业：`16997938`；申请 1×H200、8 CPU、180 GB RAM、1 小时，运行节点 `gh105`。
- 环境：`/scratch/gz2522/gz2522/venvs/bio-proassist-py312`；PyTorch 2.5.1+cu121、
  Transformers 4.49.0、Accelerate 1.3.0、NumPy 1.26.4、Decord 0.6.0。
- 3B 缓存 revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`。
- 32B 缓存 revision：`7cfb30d71a1f4f49a57592323337a4a4727301da`，只读复用本机可访问缓存。
- 设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不下载模型、不修改既有 Python 环境。

原始输出：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_16997938/`。
日志：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_16997938.{out,err}`。

输出文件：

| 文件 | 内容 |
|---|---|
| `run.json` | 时间、版本、硬件、模型路径、设备分布、环境、参数和源文件哈希 |
| `inputs.json` | 每个决策的采样帧、保留帧、时间戳 |
| `gold.predictions.jsonl` / `rollout.predictions.jsonl` | 每次原始输出、指导、输入提示词、历史、参考答案和耗时 |
| `gold.summary.json` / `rollout.summary.json` | 实时累计指标及是否完成 |
| `summary.json` | 两种历史设置的最终指标 |

## 指标口径

- gate Accuracy：25 个决策中预测标签正确的比例，invalid 计错。
- interrupt/silent 各自 Precision、Recall、F1；Macro F1 是两类 F1 算术均值；
  G-mean F1 是两类 F1 的几何均值。
- 混淆矩阵以真实标签为行、预测标签为列，显式保留 invalid 列。
- 分别统计 error / recovery / protocol 阶段的正确数和触发次数。
- 大模型调用次数、节省调用次数、非空指导数。非空只衡量是否产出文字，**不代表语义正确**。
- 原始指导和参考话术并列供人工审核；没有计算未经校准的 LLM judge 或把门控准确率称为指导准确率。
- `judger_generate` / `expert_generate` 是模型包装器内 generate 计时，不含 processor。
  `decision_wall` 包括视频解码/筛帧、processor 和两个模型生成；数据预处理一次后复用，
  各模式的决策时间均计入同一实际预处理成本。模型加载另计，未做 warmup。
- 没有 always-32B 基线实测，因此不声称墙钟加速比；调用减少比例不能直接等同速度提升。

## 实测结果

两种历史设置均完成 25/25 决策。Slurm 状态 `COMPLETED`、退出码 `0:0`，总墙钟 5 分 13 秒。
模型全部在单张 H200 上，无 CPU/disk offload；模型加载 154.47 秒，视频预处理 47.47 秒。

| 历史设置 | 正确数 / Accuracy | Macro F1 | G-mean F1 | interrupt P / R / F1 | silent F1 | 大模型调用 | 非空指导 |
|---|---|---|---|---|---|---|---|
| gold | 13/25 · 52% | 0.3421 | 0.0000 | 0.5417 / 0.9286 / 0.6842 | 0.0000 | 24/25 | 24 |
| rollout | 11/25 · 44% | 0.3056 | 0.0000 | 0.0000 / 0.0000 / 0.0000 | 0.6111 | 0/25 | 0 |

两种设置均无格式错误。`gold` 的 11 个真实 silent 全部被误触发；仅 d01 预测 silent，但 d01 应 interrupt。
`rollout` 全部预测 silent，漏掉全部 14 次应有干预，包括 6 个错误事件。

| 设置 / 真实标签 | 预测 interrupt | 预测 silent |
|---|---:|---:|
| gold / interrupt | 13 | 1 |
| gold / silent | 11 | 0 |
| rollout / interrupt | 0 | 14 |
| rollout / silent | 0 | 11 |

| 阶段 | gold 正确数 | rollout 正确数 |
|---|---:|---:|
| error | 6/6 | 0/6 |
| recovery | 0/6 | 6/6 |
| protocol | 7/13 | 5/13 |

| 耗时指标（秒/决策） | gold | rollout |
|---|---:|---:|
| 小模型 generate | 0.471 | 0.342 |
| 触发时大模型 generate | 1.204 | 未调用 |
| 含视频处理/processor 的决策时间 | 3.804 | 2.380 |

平均输入 15.68 帧，重投影后保留 12.64 帧。gold 仅减少 1/25（4%）大模型调用；
rollout 虽然避免全部调用，但同时漏掉全部干预，不能视为有效省算力方案。

固定标签的算术参照（不调用模型）：always-interrupt Accuracy=56%、Macro F1=0.3590；
always-silent Accuracy=44%、Macro F1=0.3056，两者 G-mean F1 均为 0。
本次 gold 低于 always-interrupt 的 Accuracy，rollout 恰好退化为 always-silent。

### 指导语句与失败案例

所有 50 个决策及其中 24 条实际大模型指导已保存为 [逐决策表](decisions.md) / [CSV](decisions.csv)。
以下是根据当前参考标注进行的定性对照，未作为专家审核的语义准确率计分：

- **d05（未戴手套）**：gold 门控触发，但大模型说 “Turn off the UV light…”；未针对手套问题给出指导。
- **d07（前窗高度过高）**：gold 门控触发，大模型要求给培养瓶外表消毒；未针对前窗高度纠正。
- **d10（培养瓶干置）**：参考要求立即加 PBS，大模型却指导加完全培养基中和胰酶，步骤定位错误。
- **d18（孵育完成）**：大模型生成加完全培养基中和胰酶的指导，与该点参考主要意图一致。
- **d19（胰酶暴露过久）**：门控触发，但指导轻柔吹打分散细胞，没有要求及时中和胰酶。
- **全部 6 个 recovery**：gold 仍继续打断，反映修复后停止提示的能力不足。

这次实验说明给定零样本提示下的 3B 门控对历史设置非常敏感，且两种设置都出现单类偏置；
24/25 次调用或 6/6 错误点触发并不意味着识别了正确错误类型。具体成因仍需进一步消融验证。
后续可独立比较无历史/去掉历史决策标记/仅当前窗口，并在独立开发集上训练门控；本次没有针对这 25 个标签反复调提示词。

### 结果复核与留档

- 重算全部预测的指标，逐项核对已保存汇总；Macro/G-mean F1 与本地官方 Wearable-AI starter kit 的 `binary_metrics` 一致。
- 检查全部 50 次路由：expert 输出/提示词存在当且仅当小模型判 interrupt；silent 时 expert 原始输出为空。
- 检查两种模式的完整历史、参考答案对齐、无未来帧，以及运行源代码哈希均一致。
- [summary.json](summary.json)、[run.json](run.json)、[inputs.json](inputs.json) 为完成作业的原始文件副本。
- 完整逐次提示词、原始输出和 Slurm 日志保留在上述 scratch 路径。

## 局限

仅一个有剪辑切换的 draft 视频、25 个相关决策点，不能代表跨实验或真实连续佩戴场景的泛化能力。
静态 SOP 与评估数据来自同一文档，属于给定 protocol 的任务内零样本评估。
gold 模式使用真实既往指导，有历史信息优势；主要部署参考应同时查看 rollout。
本实验记录模型原始建议，不把生成建议当作经过领域专家确认的操作指令。
