# EgoProactive-Bio：原生 LabGate 流程适配与复测

> 本目录保留原生端到端运行的原始证据。数据、输入输出、解耦试验和总体结论见
> [`../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。
> 本运行遗漏了 `dialog[i]` 和 `task[i]` 模型输入，现仅作为无上下文视觉消融；修正版见
> [`../egoproactive_bio_contextual_20260906/`](../egoproactive_bio_contextual_20260906/README.md)。

## 目的

修正前两版偏离原项目流程的问题，以尽量少的改动使用 LabGate 原生级联：

```text
16 个相邻原始帧
  → 原 reproject_keep(tau=0.12)
  → Qwen2.5-VL-3B + 原 YES/NO reason prompt
  → 仅 YES 调 Qwen2.5-VL-32B + 原 TYPE/MSG prompt
```

Bio 任务比原 LabGate 多一类“正常步骤完成后提示下一步”。因此只做两个必要扩展：

1. 小模型 reason enum 增加 `next_step`，并增加一条对应触发规则。
2. 大模型仍使用 `SAFETY / ACTION_ERROR / ASSISTANT / NONE`，其中 next_step 映射为 ASSISTANT。

没有使用 v1 的 `$interrupt$/$silent$` prompt、gold/rollout 历史，也没有使用 v2 的 JSON、
system prompt、预测状态、历史图片、规则 2fps 视频或提高后的像素预算。

## 数据标注适配

原始 `EgoProactive-Bio` 文件保持不变。`prepare_egoproactive_bio_labgate.py` 读取已有 compact annotation
和 composition，生成原 `eval_zeroshot.py` 风格的 JSONL：

| 原标注 | LabGate gt_type | 小模型期望 reason | 数量 |
|---|---|---|---:|
| `$silent$` | none | none | 11 |
| error，UV 开启/未戴手套 | safety | safety | 2 |
| 其他 error | action_error | action_error | 4 |
| 其他 `$interrupt$` | assistant | next_step | 8 |

模型只接收 `video/frame_indices/protocol/asr_text`。本视频没有音轨，全部 `asr_text` 为空。
`gt_type/gt_reason/reference_guidance/phase/pair_id/task/interval` 仅用于输出后评分和审核。

每个决策窗口使用原 `prepare_eval.window_indices` 产生 16 张相邻帧，随后由原 pipeline 重投影筛选。
这严格复现原路径，也保留了它对持续时间动作可能不利的限制。

## 执行

```bash
cd /home/gz2522/VLM4WetExperiment-reference
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  -m unittest discover -s LabGate -p 'test_*.py' -v
sbatch LabGate/run_egoproactive_bio_native.sbatch
sbatch LabGate/run_diagnose_bio_native.sbatch
```

- 14 项测试通过，包括 25 条映射、模型字段隔离、原 pipeline 条件路由和 reason 传递。
- 两级评估作业：`16999805`，H200，`COMPLETED 0:0`，4 分 41 秒，包含 always-expert 对照。
- 最终 3B-only 诊断作业：`17000307`，L40S，`COMPLETED 0:0`，1 分 59 秒。
- 初步诊断 `16999806` 已被 `17000307` 的六组完整消融覆盖，不混入最终表格。
- 适配数据：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/data/egoproactive_bio_native_20260905.jsonl`。
- 两级输出：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_native_16999805/`。
- 诊断输出：`/scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/outputs/bio_native_diag_17000307/`。
- 正式大小模型 prompt 见 [`PROMPTS_BIO_NATIVE.md`](../../PROMPTS_BIO_NATIVE.md)。

## 为什么小模型总是 silent：诊断设计

对同一批 25 个决策窗口做六组 3B 推理；除去重投影变体外，其余都使用同一批筛选后画面：

| 变体 | 目的 |
|---|---|
| native_original_prompt | 原 LabGate prompt；确认原任务定义在无 ASR Bio 画面上的行为 |
| native_bio_prompt | 只增加 next_step 规则；这是正式 Bio 门控 prompt |
| native_bio_no_reprojection | 正式 prompt，但保留原 16 帧；隔离筛帧的影响 |
| native_bio_black | 同一正式 prompt，画面改成全黑；测试输出是否需要当前视觉证据 |
| native_bio_explicit_text_probe | 保留真画面，用明显文字描述安全/错误/已完成；测试模型能否按 prompt 触发 |
| native_original_question_probe | 原 prompt + 原项目示例问题；检查标准 user_query 路径 |

显式文字使用 GT 类别构造，只是诊断模型是否能触发，绝不计入正式准确率，也不传给正式大模型。
黑画面是分布外极端对照，只能说明视觉必要性，不能代表正常泛化。

## 正式两级结果

25 个决策点均完成。标签中 14 个应触发、11 个应静默；3B 只触发 1 次。

| 指标 | 结果 |
|---|---:|
| Gate Accuracy | 48.0% |
| Gate Precision / Recall / F1 | 1.000 / 0.071 / 0.133 |
| TP / FP / FN / TN | 1 / 0 / 13 / 11 |
| 大模型调用 | 1/25（4%） |
| 级联 TYPE Accuracy | 48.0% |
| Always-expert TYPE Accuracy | 20.0% |
| 平均输入帧 / 筛选后帧 | 16.00 / 2.24 |
| 级联生成延迟 / always-expert | 0.282 s / 1.718 s |

四类混淆为：8 个 `assistant` 全变成 `none`，4 个 `action_error` 全变成
`none`，2 个 `safety` 中识别 1 个，11 个 `none` 全部正确。48% 只比恒定预测
`none` 的 44% 高 4 个百分点，不能将 6.10× 调用时间比当作有效加速，因为它来自
13 个漏报。

唯一触发点 `d03` 是 UV 开启：3B 输出 `YES. reason=safety`，32B 输出
`TYPE: SAFETY`，指导语为：

> The UV light appears to be on, which is unsafe for working inside the biosafety cabinet. Ensure the UV light is turned off before proceeding.

同一样本的 always-expert 对照没有收到小模型 reason，输出 `TYPE: NONE`。这一个样本证明
reason 传递能实际改变大模型结果，但单个样本不足以估计整体改善。Always-expert
20% 是 TYPE 准确率，不是指导语语义正确率；本次没有专家对 25 条文本指导进行评审。

## 3B 消融诊断

| 变体 | 触发 | TP / FP | 解读 |
|---|---:|---:|---|
| native_original_prompt | 0/25 | 0 / 0 | 原 LabGate prompt 在无 ASR Bio 画面上全 NO |
| native_bio_prompt | 0/25 | 0 / 0 | 只加 next_step 仍全 NO |
| native_bio_no_reprojection | 1/25 | 1 / 0 | 保留 16 帧只找回 `d23`，且 reason 错判为 action_error |
| native_bio_black | 0/25 | 0 / 0 | 黑帧不会凭 prompt 触发 |
| native_bio_explicit_text_probe | 7/25 | 7 / 0 | 把 GT 类别写成明确文字后，14 个正例仍漏 7 个 |
| native_original_question_probe | 2/25 | 1 / 1 | 原项目示例问句也未稳定触发 |

显式文字探针找到了 2/2 个 safety、3/4 个 action_error，另一个 action_error 以
`user_query` 触发；8 个 next_step 只触发 1 个。这证明模型能生成合法 YES，
严格解析器也没有吞掉 YES，但即使给它直接文本线索，`next_step` 规则也很弱。

正式 H200 作业的 `d03` 是 YES，L40S 诊断中同一 prompt 和帧为 NO。两边都是
`do_sample=False` 的 bfloat16 推理，因此这不是采样随机性；结果表明该样本在不同
GPU 数值路径下处于决策边界附近。正式分数保留 H200 原运行结果，诊断表保留
L40S 原输出，不挑选较好的一次合并。

## 为什么几乎总是 silent

1. **时序证据被大量删除。** 原重投影筛选后平均仅 2.24/16 帧；25 个窗口中
   14 个只剩 1 帧（包装器为满足 video 输入会复制成 2 帧）。“步骤刚完成”、液体是否被移除、
   是否持续暴露都需要动态证据。去掉筛选仅恢复 1 个触发，所以这是因素之一，不是唯一原因。
2. **Bio 没有 ASR，而原 LabGate 评估很依赖文字。** 原 42-case 构建中有 25 条手工 ASR
   直接写出问题、危险或自报错误；Bio 正式输入全是 `(no user speech)`。因此原 README 的
   76.2% 不能平移为这里的纯视觉预期。
3. **静态 SOP 没有当前步骤状态。** 短窗口加整份 SOP 不能告诉模型之前的合法步骤是否完成。
   特别是 next_step，需要稳定的 protocol progress 跟踪，而原 LabGate 只是无状态短窗口 gate。
4. **3B 的视觉粒度和规则跟随不足。** 每帧上限 100,352 pixels，手套、UV 指示、瓶内液体、
   微小器材和动作顺序很难从第一人称画面稳定区分。标签知情的文字探针也只找回一半正例，
   说明不能只通过再加一句 prompt 解决。
5. **v2 额外有输出结构退化，但不是根因。** v2 的 JSON/状态 prompt 反复复制同一 observation；
   回到原生 YES/NO pipeline 后格式正常，仍然 24/25 silent。

因此，小模型“总是 silent”的主因不是没有 prompt、没有把 reason 传给大模型，
也不是解析错误。当前零样本原生 gate 的输入表示无法稳定支撑这些细粒度、强时序的 Bio 标签。

## 下一步设计含义

不建议再用 GT 文字塞进 ASR 追分；那只是标签泄漏。更合理的方向是先单独评估小模型对
UV/手套/液体/动作的可视性，再引入不依赖 GT 的步骤状态机或针对 Bio 的监督训练。对强时序类别
保留规则采样帧，对视角重复帧另做消融，而不是默认把 16 帧压到 1–2 帧。

## 复核与留档

已逐行独立复算指标，检查 25 条样本唯一性、YES/NO 解析与路由一致性、NO 时 expert
零调用、YES 时 reason 确实出现在 expert prompt，以及 GT/参考指导没有进入模型 prompt。
诊断文件含 6×25=150 条原始输出。

- [summary.json](summary.json) / [run.json](run.json) / [25 条正式预测](predictions.jsonl) /
  [精简逐决策表](decisions.csv)
- [diagnostic_summary.json](diagnostic_summary.json) / [150 条诊断输出](diagnostic_predictions.jsonl)
- [适配后输入](inputs.jsonl) / [适配元数据](inputs.meta.json)
- `source/` 保存两个正式作业实际使用的关键源码快照。
