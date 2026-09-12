# EgoProactive-Bio v2：理由传递与输入修正

> 这是 v2 失败路径留档，不是当前推荐流程。统一结论和最终输入输出定义见
> [`../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。

本次按用户确认补齐两级模型理由传递，并落实上一轮定位出的输入问题。
仓库其他目录的核查见 [REPOSITORY_AUDIT.md](../../REPOSITORY_AUDIT.md)，
完整可读 prompt 见 [PROMPTS_BIO_V2.md](../../PROMPTS_BIO_V2.md)。

## 运行配置

- 同一 EgoProactive-Bio `0.3.0-draft`，一个视频、25 个决策点；gold/rollout 各完整运行一次。
- 同一 3B、32B 基础模型和权重 revision；未使用相邻项目已有但表现退化的 LoRA。
- 小模型输出 JSON：decision、reason_type、observation、current_step、step_status。
- reason_type：safety_warning / action_error / next_step / user_query / none。
- 大模型仅在小模型 interrupt/yes 时调用，并收到实际的小模型理由、当前观察、步骤估计；
  prompt 要求将其视作未验证假设，用画面核实，不能无条件照搬。
- 独立 system prompt，历史使用实际 chat 角色；去掉历史中的控制标签，保留最近最多四条对话。
- 保存最近两次小模型预测的观察/步骤状态，包括 silent 决策；不使用当前真实步骤标签。
- 当前窗口按规则时间间隔采样，默认 2 fps，最长 32 帧，偶数帧，末帧严格早于决策时刻。
- 当前帧不做重投影删除；最多四张历史画面作为独立图片输入，附带各自时间戳。
- fps 显式进入 processor，记录并核验 `second_per_grid_ts`。
- 像素预算 200,704/帧（v1 为 100,352）；小模型最多 160 token，大模型最多 128 token；greedy decoding。
- v1 结果仍在 [原实验目录](../egoproactive_bio_20260905/README.md)，没有覆盖。

这是多个修正合在一起的系统版本比较，不是单独衡量“增加理由”的因果消融。
新增步骤状态是模型预测的短期记录，不是已验证的动作识别器/完整 protocol 状态机；错误状态可能传播。
去掉历史标签也不能消除历史语义与对话位置的影响。

## 执行记录

```bash
cd /home/gz2522/VLM4WetExperiment-reference
python -m unittest discover -s LabGate -p 'test_*.py' -v
sbatch LabGate/run_egoproactive_bio_v2.sbatch
```

- 10 项测试通过；额外用假模型实测原 `LabGate.run_frames`：safety/action_error 进入 expert prompt，NO 时零调用。
- Slurm 作业 `16999592`，H200，节点以运行元数据为准，申请 8 CPU、180 GB RAM、40 分钟。
- 输出：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_v2_16999592/`。
- 日志：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/logs/bio_v2_16999592.{out,err}`。
- 模型与数据路径从 v1 的 `run.json` 读取，并要求标注 SHA-256 与 v1 一致。
- `inputs.json` 保存当前/历史索引、fps 和时间戳；预测 JSONL 保存所有实际 system/user prompt、
  清理后 history、此前状态、小模型结构化/原始输出、大模型指导、processor 网格与时间编码统计。
- 语义指导准确率和理由类型准确率尚未建立领域专家审核标准；报告原始输出和非空数量，不把它们称为正确指导率。

## 格式错误尝试与修复

首个 v2 尝试作业 `16999479` 在核对原文时发现模型照抄 `decision="interrupt|silent"`，
而兼容前缀解析把它当成 interrupt。该次运行已主动取消，全部部分分数作废；原始输出及当时源文件快照保留在
`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_v2_16999479/`。

修复后 JSON decision 必须与枚举值完全一致；包含多个选项、列表或空值都记 invalid，不能调用大模型。
新增实际路由回归测试覆盖这些情形；system prompt 改成逐字段要求，不再给出容易照抄的 `a|b` 模板对象。
随后提交新的完整作业 `16999592`，不复用失效预测，也不将 schema 修复称为模型准确率优化。

## 结果

完整作业 `16999592`：`COMPLETED`，退出码 `0:0`，节点 `gh119`，总墙钟 3 分 41 秒。
gold/rollout 均完成 25/25 决策，所有模型加载在 GPU 0，无 CPU/disk offload。

| 版本 / 历史 | Accuracy | Macro F1 | G-mean F1 | invalid | 大模型调用 |
|---|---:|---:|---:|---:|---:|
| v1 / gold | 52% | 0.3421 | 0.0000 | 0 | 24/25 |
| v1 / rollout | 44% | 0.3056 | 0.0000 | 0 | 0/25 |
| v2 / gold | 36% | 0.2727 | 0.0000 | 3 | 0/25 |
| v2 / rollout | 44% | 0.3056 | 0.0000 | 0 | 0/25 |

**本次没有得到门控能力提升。** v2 gold 为 22 次 silent、3 次 invalid，rollout 为 25 次 silent；
两种模式的 interrupt recall 均为 0，6 个错误事件均未触发。

gold 的 d04、d16 输出了冒号分行字段而非 JSON；d13 输出自由文本 Action Error，
均按事先声明的严格协议记 invalid、计错且不调用大模型。完整 schema 有效数为 gold 22/25、rollout 25/25。

更明显的问题是，所有解析成功的 47 次小模型观察都重复：
`The person is wearing gloves and holding a tube.`，current_step/step_status 均为 unknown。
这显示当前版本发生了观察/状态复制退化；无法仅凭这一合并修改实验确定是模型视觉能力、提示词、
历史/预测状态干扰还是多模态输入方式各自贡献了多少。schema 有效不代表观察正确。

**理由传递已通过路由与 prompt 测试，但最终正式 v2 因没有有效 interrupt，没有调用大模型，
因此不能声称新理由已在这次正式运行中改善了指导，也没有 v2 指导语义分数。**
被取消的首个尝试中的大模型调用来自无效解析，不作为有效大模型评估结果。

### 输入与运行复核

- 实际 processor 的图像/视频混合输入、chat roles 和 `second_per_grid_ts` CPU smoke 通过。
- 50 次实际小模型输入的时间网格都等于 `2/fps=1.0`；d10/d14 当前帧均为 16，历史画面独立输入。
- 逐行复算指标，验证参考答案、历史清理、独立模式状态、无未来帧、严格路由和所有关键源文件 SHA-256。
- 有效 JSON 的理由字段全部为 none；没有生成可验证的 safety_warning/action_error 触发理由。
- 下一步应先用独立观察/分类任务验证小模型是否看懂当前视频，再测试理由输出与短期状态，
  而不是继续在这 25 点上调提示词来追求分数。

### 留档

[summary.json](summary.json)、[run.json](run.json)、[inputs.json](inputs.json)、
[逐决策 CSV](decisions.csv)、[gold 完整预测与 prompt](gold.predictions.jsonl)、
[rollout 完整预测与 prompt](rollout.predictions.jsonl)。
本目录 `source/` 保存最终正式运行的关键代码快照；首次失效尝试示例在
[failed_attempt_16999479.json](failed_attempt_16999479.json)。
