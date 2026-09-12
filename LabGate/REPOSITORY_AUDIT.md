# Prompt 与仓库组件核查（2026-09-05）

> **2026-09-06 修正：** 后续复核确认，原生 Bio 复测虽然调用了 3B/32B prompt，却错误地把
> `dialog[i]` 和 `task[i]` 排除在模型输入之外，且给 32B 的 trigger JSON 留下了空
> `current_step`。这不是数据集本身缺字段，而是适配错误。修正版按 WearableAI 的高层 query +
> 最近 4 个历史 turn 方式把 history 作为真实 chat messages 传入，并把同一个 `task[i]` 作为
> `current_step` 传给 3B、32B 及 32B trigger JSON。完整累计 history 另作消融。详见
> [上下文版输入与 prompts](PROMPTS_BIO_CONTEXTUAL.md)。

## 直接结论

两个模型原本都有 prompt；缺的是明确的信息传递和部分已有实现经验的复用。

| 路径 | 小模型输入/输出 | 大模型输入/输出 | 查到的连接问题 |
|---|---|---|---|
| 原版 `LabGate/prompts.py`、`pipeline.py` | `judger_prompt`，输出 `YES. reason=safety/action_error/user_query` 或 `NO` | `expert_prompt`，输出 TYPE + MSG | 原 pipeline 解析 reason 后仅记录到 `GateResult`，没有传给 expert_prompt；本次已接通 |
| 第一版 Bio `egoproactive_bio.py` | `gate_prompt`，纯 `$interrupt$/$silent$` | `guidance_prompt`，只知道已触发 | 适配时丢掉原来的 reason；此次保留 v1 作为基线，以 v2 补齐 |
| Bio v2 `proactive_v2.py`、`proactive_models.py` | 单独 system prompt + 清理标签后的 chat history + SOP/画面/历史观察；输出结构化决策、理由、观察、步骤/状态 | 独立 system prompt + 相同视觉上下文 + 实际小模型 reason/observation/current_step | 条件调用与理由传递已实现，每次 system/user prompt 和实际 processor 输入统计均留档 |

第一版 Bio 的提示词确实通过 `QwenVL.generate` 写进最终 chat template，
不是“定义了函数但没有使用”。不过该包装器把整个提示词及历史扁平放进一个 user 消息，
没有使用原 WearableAI 的独立 system/assistant 消息结构。

## 仓库内其他目录

| 目录 | 实际功能 | 与本任务的关系/处理 |
|---|---|---|
| `WearableAI/` | 官方风格主动辅助 system prompt、真实 chat history 角色、最多四轮历史、累计画面、LoRA 训练/评估 | 修正版上下文评估复用其“高层 query + 最近四个 turn”结构；原始控制标签和清理标签两种条件分别测试 |
| `FineBioQwenStream/` | 7B LoRA，CONTINUE/HALT + missing_protocol/wrong_execution + 简短理由 | 有“类别+理由”的输出设计；但标签是整段完整 protocol 的缺失/错配，不是细胞传代内的具体动作/安全标签，不能直接当 Bio 已训练 gate |
| `FineBioWhen2See/` | ORB/homography 重投影差异筛帧 | 不是语义 VLM、没有大小模型 prompt。v1 已复用，但会删除当前窗口的时间证据；v2 当前视频不再应用它 |
| `ExpVid/` | 图像→caption SSL、图像/旁白+MCQ 评估 | 不是已实现的 silent/interrupt 两级模型；caption 可用于未来观察监督，不能把其 MCQ 准确率当 Bio 门控效果 |
| 根目录旧 LLaVA 脚本 | protocol compliance prompt、LoRA、视频推理 | 模型/任务不同，非遗漏的即插即用 Qwen3B→32B 门控 |
| `LabGate/protocols.py` | 七类 FineBio protocol 卡片 | 不含 cell passaging；Bio 的静态 SOP 必须从本次用户数据补充。v1 已做，v2 复用 |
| `LabGate/asr.py` | 文本 ASR 直传；可选 Whisper | Bio 视频无音轨，不应把标注的错误说明伪装成 ASR 来提高成绩 |
| `LabGate/prepare_eval.py`、`examples/ASR_TEXT_EXAMPLES.md` | 为正确画面配人工填写的提问/危险/自报错误文字 | 解释原 42-case 成绩与这次纯视觉困难度不一致，见下文 |

附带发现：`FineBioQwenStream/infer_stream.py` 用三个实参调用 `build_user_prompt`，
而现有 `protocol_prompt.py` 定义只接受两个参数，说明该旧入口还存在接口未同步。
它不在本次 Bio 运行路径内；本次未修改该无关入口，也没有把它当成已验证的直接替换方案。

## 本机相邻项目（不在这个 GitHub 仓库内）

| 本地路径 | 已有内容 | 本次发现 |
|---|---|---|
| `/home/gz2522/bio-proassist/src/wearableai_protocol.py` | 独立 system/history、累计图像输入 | 已有相同任务的协议实现经验，首次适配未充分查阅 |
| `/home/gz2522/bio-proassist/data/prepare_dataset.py` | 较早 current-clip 分支，明确 `_clean_dialog_text` 去掉历史控制标签 | 历史标签清理实际上已有先例，这次 v2 已补上等价处理 |
| `/home/gz2522/bio-proassist/WEARABLEAI_PROTOCOL_FINAL_REPORT.md` | 2026-08-22 完整报告，105 视频/1,481 决策；LoRA 只在首点触发，后续 1,376 点全 silent | 是应提前阅读的重要负结果，提示历史/位置捷径早已出现；不能把现成 adapter 当成已验证有效的修复 |
| `/scratch/gz2522/bio-proassist/outputs/.../best/` | 确实存在 3B LoRA adapter，非仅有训练脚本 | 当前实验明确使用基础模型、没有载入 LoRA；上述负结果是没有直接替换为该 adapter 的依据。以后可作为单独对照 |
| `/home/gz2522/gz2522/Bio_Agent/Judger/` 及 `VLM_Reasoner/Judger/` | 感知/运动/HOI/融合、运行时状态 `JudgerContext`、冷却一致性、触发理由 | 是低层视觉触发原型，非另一个已训练语义小 VLM；README 明确 Phase 2 未实现，含 mock 支持。其冷却状态不能替代 protocol 步骤状态，也不能不加区分地抑制新的危险 |
| `/home/gz2522/bio-agent-exp/02_smallVLM_test/` | 空目录 | 没有遗漏的模型或 prompt 文件 |

## 原 LabGate 42-case 评估的文本条件

按 `prepare_eval.py` 的完整案例清单（假设视频均存在）计数：

- 10 个 NONE：正常视频，空 ASR。
- 10 个 ASSISTANT：同组正常视频，手工提问文字。
- 10 个 SAFETY：同组正常视频，手工描述危险。
- 5 个 ACTION_ERROR：正常视频，手工自报操作错误。
- 3 个 ACTION_ERROR：真实错误片段，空 ASR。
- 4 个 ACTION_ERROR：视频与指定 protocol 错配，空 ASR。

即 25/42 个案例带有直接描述提问或问题的文字。代码会跳过不存在的视频，
上述是构建清单的预期构成，不是对原作者训练结果文件的重新验证。
README 的 76.2% 是另一任务的既有报告，本次没有重跑，不能直接拿来期待无音频 Bio 的相同成绩。

## 本次补齐的 v2 链路

小模型对外仍决定 interrupt/silent，同时内部返回：

```json
{
  "decision": "interrupt",
  "reason_type": "safety_warning",
  "observation": "brief description of the current visible issue",
  "current_step": "observed action or unknown",
  "step_status": "in_progress"
}
```

上例是接口示意，不是本次模型实测输出。
reason_type 包括 `safety_warning / action_error / next_step / user_query / none`。
加入 next_step 是因为 Bio 的 interrupt 不只包含错误，还包括正常步骤完成后的指导。

1. 大模型仅在小模型 interrupt/yes 时调用；理由/观察/步骤来自小模型实际输出，绝不从当前 GT 取值。
2. expert prompt 明确标记触发信息是未验证假设，要求用当前画面核对；不把小模型理由当事实。
3. JSON decision 进行完整枚举匹配，`interrupt|silent` 等歧义字符串记 invalid、不调用大模型；小模型 JSON 解析失败记录 `schema_valid=false`；兼容合法旧二分类词，缺失理由记 unknown，不能凭空填造。
4. 历史去控制标签、保留真正 chat 角色；小模型 silent 时也记录观察与状态，最多保存两次既往观察。
5. 当前窗口规则采样且不做重投影删除，实际 fps 传入 processor 并验证 `second_per_grid_ts`。
6. 最多四张历史图片独立输入，带时间戳；不与当前 video 混成一个等间距序列。
7. 默认像素预算从 100,352 增至 200,704；所有变更组成一个 v2 系统版本，不能将分数变化只归功于理由字段。

完整可读提示词见 [PROMPTS_BIO_V2.md](PROMPTS_BIO_V2.md)，代码见
[proactive_v2.py](proactive_v2.py)、[proactive_models.py](proactive_models.py)、
[eval_egoproactive_bio_v2.py](eval_egoproactive_bio_v2.py)。

最终 v2 作业 `16999592` 已完成 50 个决策调用：理由传递路径测试通过，但真实小模型未产生有效
interrupt，正式运行没有调用大模型。结果为 gold 36% / rollout 44% Accuracy，不能宣称
补齐字段提高了识别或指导质量。详见 [完整实测记录](experiments/egoproactive_bio_v2_20260905/README.md)。

## 原生流程复测结论

为排除 v1/v2 适配自身引入的偏差，最终复测回到原 `LabGate` 路径：16 个相邻帧、
`reproject_keep(tau=0.12)`、3B YES/NO reason prompt，仅 YES 调用 32B TYPE/MSG prompt。
只增加 Bio 必需的 `next_step` 规则，并将小模型理由标为未验证信息传给大模型。

实测触发 1/25，Recall 7.1%。原生格式没有 v2 的 JSON 退化，但仍然几乎全 NO；
因此 v2 prompt 是附加问题，不是根因。重投影平均将 16 帧压到 2.24 帧，去掉它也只多触发
1 条。标签知情的显式文字探针只触发 7/14 正例，说明解析器没有吞 YES，而是
3B 对纯视觉 Bio 动作、时序步骤和 `next_step` 规则的零样本支持很弱。详见
[原生实验记录](experiments/egoproactive_bio_native_20260905/README.md)。
