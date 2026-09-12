# EgoProactive-Bio × LabGate 完整实验报告

**实验日期：** 2026-09-05 至 2026-09-06  
**仓库基线：** `Nemo0412/VLM4WetExperiment` commit `5755752dfd51d1cc1260610b656a27a42be83781`  
**任务：** Qwen2.5-VL-3B 判断是否触发；触发后 Qwen2.5-VL-32B 输出实验指导。  
**主结果目录：** [`experiments/egoproactive_bio_contextual_20260906/`](experiments/egoproactive_bio_contextual_20260906/)

## 0. 重要修正

最早的“原生 LabGate”适配有两个输入错误：

1. 数据集的 `dialog[i]` 保存了第 `i` 轮之前的对话，但当时没有传给 3B。
2. 数据集的 `task[i]` 是第 `i` 轮对齐的步骤条目，但当时 32B trigger JSON 中
   `current_step` 被留成了空字符串。

所以旧的 0–7.1% gate recall 只能作为“不含 history/task 的视觉消融”，不能代表正确的数据集适配。
修正版已经做到：

- history 作为真实 `user/assistant` chat messages 传给 Qwen；
- 相同的 `task[i]` 作为 `current_step` 同时传给 3B 和 32B；
- 32B trigger JSON 的 `current_step` 也填入完全相同的 `task[i]`；
- 32B 只在 3B 输出可解析的 `YES` 后运行，并接收 3B 实际预测的 reason；
- `answers[i]`、参考指导、GT TYPE 和 GT reason 仍只用于生成后的评分。

完整 prompt 和 d03 输入示例见 [`PROMPTS_BIO_CONTEXTUAL.md`](PROMPTS_BIO_CONTEXTUAL.md)。旧输入保留在
[`PROMPTS_BIO_NATIVE.md`](PROMPTS_BIO_NATIVE.md)，并已标明是视觉消融。

## 1. 主要结论

### 1.1 真实级联

最接近现有 WearableAI/LabGate 设计的主条件是：高层 query + 最近 4 个原始历史 turn、保留历史
`$interrupt$` 标记、原 LabGate 重投影筛帧、相同 `task[i]` 传给两个模型。

| 指标 | 主条件结果 |
|---|---:|
| 决策点 | 25（14 interrupt / 11 silent） |
| 3B TP / FP / FN / TN | 5 / 4 / 9 / 7 |
| Gate accuracy | 48.0% |
| Gate precision / recall / F1 | 55.6% / 35.7% / 0.435 |
| 3B 触发率 | 36.0%（9/25） |
| 正例 reason accuracy | 7.1%（1/14） |
| 32B 在已路由真阳性上的 TYPE accuracy | 60.0%（3/5） |
| 全 25 点端到端 TYPE accuracy | 40.0%（10/25） |
| 14 个正例端到端 TYPE 成功 | 21.4%（3/14） |
| 14 个正例端到端语义完整指导 | 28.6%（4/14） |
| 人工复核：5 个已路由真阳性的指导 | 4 correct / 1 partial |
| 人工复核：4 个误触发 | 4 个均为不必要或错误指导 |

补回 context 后，3B 不再恒定 silent，但仍不是可靠 gate。它漏掉 9/14 个应触发点，并在 4 个
silent 点调用 32B。32B 能利用 `current_step` 挽救部分错误 reason：例如 d03 的 3B reason 错为
`next_step`，32B 仍输出正确的 UV 安全警告；但它不能稳定拒绝误触发，d04/d06 会重复已经解决的
UV 警告。

### 1.2 解耦结论

- **3B：** 是否保留历史控制标记对触发率影响很大。final timing prompt 下，官方 raw history 的
  Recall 是 35.7%，clean history + 16 帧的 Recall 只有 7.1%。说明 raw `$interrupt$` 提供了很强
  的触发先验；去掉它后，3B 又接近全 silent。
- **32B：** 在 14 个正例上强制正确触发并提供 oracle coarse reason 后，官方 raw history +
  重投影的 TYPE accuracy 为 64.3%，自动语义完整率为 50.0%，人工严格结果为
  6 correct / 4 partial / 4 wrong。它已明显优于没有 current step 时的 1–2/14，但 normal
  next-step 经常被误判成 `ACTION_ERROR`。

因此系统有三个相互独立的问题：3B 的触发时机、3B 的 reason 分类、32B 对“当前动作”和“下一动作”
的区分。给模型补齐 history/current step 是必要修复，但不足以得到可靠系统。

## 2. 数据集与标签

数据路径：

```text
/home/gz2522/bio-dataset/EgoProactive-Bio
```

| 属性 | 值 |
|---|---|
| 版本 | `0.3.0-draft` |
| 视频 | `egoproactive/val/bio_cp_oop_full_001.mp4` |
| 时长 / 规格 | 726.000 秒；960×1280；30 fps；H.264 |
| 音频 | 无 |
| 决策点 | 25 |
| 标签 | 14 `$interrupt$`，11 `$silent$` |
| 错误—恢复 pair | 6 组 |
| annotation SHA-256 | `0486d4c8091c1b406c8e2358bef5ba6f61e5ffabac3bb438a5eabe135f999968` |

原标签映射为 LabGate 评分标签：

| 数据集语义 | gt_type | 期望 3B reason | 数量 |
|---|---|---|---:|
| silent | `none` | `none` | 11 |
| UV / bare-hand error | `safety` | `safety` | 2 |
| 其他 error | `action_error` | `action_error` | 4 |
| 正常步骤结束后需要下一步指导 | `assistant` | `next_step` | 8 |

25 个对齐的 `task[i]` 如下；这些字符串在修正版中是模型输入：

| ID | 时间（秒） | 标签 | task[i] / current_step |
|---|---:|---|---|
| d01 | 0.000–3.000 | interrupt | Step 1 — Sanitize hands before handling the cell flask |
| d02 | 7.000–15.000 | silent | Step 2 — Remove the cell flask from the incubator |
| d03 | 15.000–18.000 | interrupt | Step 3 error — Do not open the biosafety cabinet while the UV light is on |
| d04 | 26.533–33.533 | silent | Step 3 recovery — Use the cabinet with the UV light off and place the disinfected flask inside |
| d05 | 33.567–41.567 | interrupt | Step 4 error — Do not reach into the sterile cabinet without gloves |
| d06 | 58.097–66.097 | silent | Step 4 recovery — Put on gloves and disinfect them with alcohol |
| d07 | 66.100–71.100 | interrupt | Step 5 error — Do not work with the cabinet sash raised too high |
| d08 | 79.197–87.197 | silent | Step 5 recovery — Set the cabinet sash to the recommended working height |
| d09 | 102.200–110.200 | interrupt | Step 6 — Remove the spent culture medium |
| d10 | 124.233–132.233 | interrupt | Step 7 error — Do not leave the flask dry after removing culture medium |
| d11 | 158.433–166.433 | silent | Step 7 recovery — Add PBS immediately after removing culture medium |
| d12 | 180.833–188.833 | interrupt | Step 8 — Discard the PBS rinse |
| d13 | 209.833–217.833 | silent | Step 9 — Add trypsin to the cell flask |
| d14 | 245.867–253.867 | interrupt | Step 10 error — Do not shake the flask vigorously after adding trypsin |
| d15 | 284.200–292.200 | silent | Step 10 recovery — Gently rock or tap the flask to distribute trypsin |
| d16 | 308.367–311.367 | interrupt | Step 11 — Disinfect the flask before returning it to the incubator |
| d17 | 314.367–322.367 | silent | Step 12 — Put the flask into the incubator |
| d18 | 421.367–429.367 | interrupt | Step 13 — Complete the intended trypsin incubation |
| d19 | 429.400–437.400 | interrupt | Step 14 error — Do not leave cells in trypsin beyond the intended incubation |
| d20 | 475.100–483.100 | silent | Step 14 recovery — Add complete growth medium to neutralize trypsin |
| d21 | 551.167–559.167 | interrupt | Step 15 — Gently pipette to disperse the cells |
| d22 | 601.167–609.167 | silent | Step 16 — Transfer cells into a new flask |
| d23 | 661.167–669.167 | interrupt | Step 17 — Add fresh culture medium |
| d24 | 695.167–703.167 | interrupt | Step 18 — Label the new flask |
| d25 | 717.167–725.167 | silent | Step 19 — Return the flask to the incubator |

`task[i]` 含有 `error`、`recovery` 和 “Do not ...” 等强语义。因此 context 版不是纯视觉评估，
分数必须与旧视觉消融分开解释。

## 3. 每轮实际输入如何构造

对第 `i` 个决策点按相同下标读取：

```text
video_intervals[i] -> 当前决策区间
task[i]            -> current_step
dialog[i]          -> 当前决策之前的 gold history
answers[i]         -> 生成结束后才用于评分
```

history 不是拼进普通文本，而是先形成真实的历史消息，再追加包含视频和当前 prompt 的 user message：

```python
messages = [
    *history,
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt},
        ],
    },
]
```

实际 Qwen processor smoke test 得到角色序列 `user → assistant → current user`，并确认渲染后的
chat template 中能找到历史文本和 current step；processor 正常产生
`pixel_values_videos/video_grid_thw/second_per_grid_ts`。

### 3.1 history 条件

| 名称 | 内容 | 用途 |
|---|---|---|
| `official_raw` | 第一条高层 query + query 后最近 4 个 turn；保留 `$interrupt$` | 主条件，匹配 WearableAI 现有实现 |
| `official_clean` | 相同 turn，移除 `$interrupt$/$silent$` | 控制标记偏置消融 |
| `full_raw` | 未截断的完整累计 dialog；保留标记 | 检验用户指出的完整历史 |
| `full_clean` | 完整累计 dialog，移除标记 | 长历史与控制标记解耦 |

最大 official history 是 5 条消息（query + 4 turn），最大 full history 是 15 条。这里使用的是数据集
gold history，不是把本次模型输出滚动写回的 autonomous rollout；这与官方 benchmark 的
teacher-forced history 方式一致。

### 3.2 d03 示例

```json
{
  "current_step": "Step 3 error — Do not open the biosafety cabinet while the UV light is on",
  "history_official_raw": [
    {"role": "user", "content": "Cell passaging"},
    {
      "role": "assistant",
      "content": "$interrupt$After sanitizing your hands, take the cell flask from the incubator."
    }
  ],
  "asr_text": "",
  "frame_indices": [487, 488, 489, 490, 491, 492, 493, 494,
                    495, 496, 497, 498, 499, 500, 501, 502]
}
```

## 4. 帧输入

每个区间中点附近取 16 个连续帧，只覆盖约 0.53 秒。原 `reproject_keep(tau=0.12)` 再根据 ORB/
homography 重投影灰度残差删除相似帧。25 个点平均从 16 帧降到 2.24 帧；14/25 个窗口只剩 1 帧。
单帧在 Qwen 包装器中复制成两帧以满足 video 输入，但不会增加时间证据。

实验同时保留 full-16 对照。它能检验重投影是否删除动作证据，但 16 个相邻帧仍不是完整视频 prefix，
也没有显式非均匀时间戳。

## 5. 两个模型的 prompt、输出和路由

### 5.1 3B Judger

3B 收到 history、视频、静态 cell-passaging SOP、空 ASR 和 `CURRENT TASK / STEP = task[i]`。
输出严格要求：

```text
YES. reason=<safety|action_error|user_query|next_step>
```

或：

```text
NO.
```

解析器只接受输出开头的完整 `YES`/`NO`。prompt 说明历史 user turn 和 current step 不是当前请求，
无 ASR 时禁止 `user_query`；final timing 版还说明“正在正确执行最近 assistant 已给出的动作时应 NO，
不要重复提醒”。

### 5.2 32B Expert

真实级联只在 3B parsed YES 后调用 32B。32B 得到与 3B 相同的 history、帧和 current step，另收到：

```json
{
  "reason_type": "<3B 实际预测 reason>",
  "observation": "",
  "current_step": "<与 3B 完全相同的 task[i]>"
}
```

`observation` 为空，因为当前 3B 接口只稳定约束 decision/reason，没有可靠观察句。reason 被明确标为
unverified，32B 应结合画面、步骤和历史复核。输出格式为：

```text
TYPE: <SAFETY|ACTION_ERROR|ASSISTANT|NONE>
MSG: <one concise instruction, or empty for NONE>
```

### 5.3 参数

| 参数 | 3B | 32B |
|---|---:|---:|
| revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` | `7cfb30d71a1f4f49a57592323337a4a4727301da` |
| dtype / attention | bfloat16 / SDPA | bfloat16 / SDPA |
| decoding | greedy | greedy |
| max new tokens | 24 | 96 |
| max/min pixels per frame | 100,352 / 3,136 | 100,352 / 3,136 |

串联作业先运行并落盘全部 3B 结果，释放 3B 权重，再加载 32B 处理 fired rows；这避免 3B+32B
同时占用 80GB H100。条件路由和传入字段与逐条交替运行相同。

## 6. 实验设计

1. **旧视觉消融：** 无 history/task，用于定位最早的 all-silent 现象。
2. **context gate 开发迭代：** 加入 task/history，观察控制标记、帧数和 timing 规则的影响。
3. **Oracle-expert 解耦：** 只取 14 个正例，假设 3B 正确触发并传入正确 coarse reason，独立测试 32B。
4. **真实条件级联：** 3B 实际输出决定是否调用 32B，转发预测 reason；两个模型均使用相同 history/task。

Oracle-expert 的 GT reason 是有意给出的能力上界。真实级联不读取 oracle reason 或参考指导。

## 7. 3B 结果与 prompt 敏感性

### 7.1 旧视觉消融

| 条件 | Recall | F1 | 说明 |
|---|---:|---:|---|
| 原生端到端重投影 | 7.1% | 0.133 | 1/25 触发 |
| 独立 gate 重投影 | 0% | 0 | 0/25 触发 |
| 独立 gate full 16 | 7.1% | 0.133 | 1/25 触发，reason 仍错 |

不同 GPU 上有一个边界样本差异，但共同结论是几乎全 silent。

### 7.2 首次补 context 的开发迭代

在还没有显式“不要把历史当当前请求/不要重复旧指导”的 prompt 中：

| 条件 | TP/FP/FN/TN | Recall | F1 | reason acc |
|---|---|---:|---:|---:|
| full raw + reprojection | 13/10/1/1 | 92.9% | 0.703 | 35.7% |
| full raw + 16 frames | 12/11/2/0 | 85.7% | 0.649 | 28.6% |

这一步证明漏掉 context 是 all-silent 的重要原因，同时暴露 raw `$interrupt$` 与历史 user turn 会把
模型推向 YES。

加入“历史不是当前 request、无 ASR 禁止 user_query”，但尚未加入 history timing 规则时：

| 条件 | TP/FP/FN/TN | Recall | F1 | reason acc |
|---|---|---:|---:|---:|
| official raw + reprojection | 11/10/3/1 | 78.6% | 0.629 | 42.9% |
| official raw + 16 frames | 13/10/1/1 | 92.9% | 0.703 | 50.0% |
| official clean + reprojection | 8/9/6/2 | 57.1% | 0.516 | 14.3% |
| official clean + 16 frames | 6/9/8/2 | 42.9% | 0.414 | 14.3% |

尽管 prompt 明确禁止，无 ASR 时仍出现 3–7 个 `reason=user_query`，说明 3B 的指令遵循也不稳定。

### 7.3 final timing prompt：真实级联中的 3B

| 条件 | TP/FP/FN/TN | Accuracy | Precision | Recall | F1 | reason acc |
|---|---|---:|---:|---:|---:|---:|
| official raw + reprojection | 5/4/9/7 | 48.0% | 55.6% | 35.7% | 0.435 | 7.1% |
| official clean + 16 frames | 1/0/13/11 | 48.0% | 100% | 7.1% | 0.133 | 0% |
| full raw + reprojection | 6/3/8/8 | 56.0% | 66.7% | 42.9% | 0.522 | 14.3% |

history timing 规则把官方 raw 的 FP 从 10 降到 4，但 TP 也从 11 降到 5；clean 条件再次退化为近乎
全 silent。所有 prompt 调整都在同一个 25 点数据上诊断开发，因此这些变化只说明行为机制，不能作为
无偏测试集增益。

## 8. Oracle-triggered 32B 结果

14 个正例都强制调用 32B，并提供正确 coarse reason、对应 history 和 `task[i]`：

| 条件 | TYPE accuracy | 语义完整率 | mean concept recall | 人工 correct/partial/wrong |
|---|---:|---:|---:|---:|
| official raw + reprojection | 64.3% | 50.0% | 69.0% | 6 / 4 / 4 |
| official raw + 16 frames | 42.9% | 28.6% | 54.8% | — |
| official clean + reprojection | 64.3% | 42.9% | 66.7% | — |
| official clean + 16 frames | 42.9% | 42.9% | 64.3% | 4 / 6 / 4 |
| full raw + reprojection | 57.1% | 42.9% | 58.3% | — |
| full clean + reprojection | 57.1% | 42.9% | 61.9% | — |

主条件按 GT TYPE 分解：

| GT TYPE | 数量 | TYPE correct | 自动语义完整 |
|---|---:|---:|---:|
| safety | 2 | 2/2 | 0/2（关键词规则低估了语义正确的 UV 文句） |
| action_error | 4 | 4/4 | 4/4 |
| assistant / next_step | 8 | 3/8 | 3/8 |

32B 已能稳定处理显式 safety/action error。例如 d03 输出关闭 UV，d07 输出降低 sash，d10 输出立即
加 PBS，d19 输出用两体积培养基中和 trypsin。主要失败集中在下一步定位：d12 重复 PBS 而不是加
trypsin，d18 有时让 flask 回 incubator 而不是中和，d23/d24 把当前步骤错当成尚未完成。

## 9. 真实级联逐条行为

主条件的 9 个 32B 调用：

| ID | GT fire | 3B reason | 32B TYPE | 指导人工判断 |
|---|---|---|---|---|
| d03 | 是 | next_step（错） | SAFETY | correct：关闭 UV 后再开柜 |
| d04 | 否 | next_step | SAFETY | wrong：恢复已完成，仍重复 UV 警告 |
| d05 | 是 | next_step（错） | SAFETY | partial：要求手套，漏掉酒精消毒 |
| d06 | 否 | user_query | SAFETY | wrong：手套恢复阶段错误重复 UV 警告 |
| d07 | 是 | next_step（错） | SAFETY | correct 内容：降低 sash；TYPE 应为 ACTION_ERROR |
| d10 | 是 | action_error | ACTION_ERROR | correct：立即加 PBS |
| d12 | 是 | user_query（错） | ACTION_ERROR | correct 内容：下一步加 trypsin；TYPE 应为 ASSISTANT |
| d13 | 否 | next_step | ACTION_ERROR | wrong：对正在正确执行的 trypsin 步骤重复提醒 |
| d25 | 否 | next_step | ACTION_ERROR | wrong：错误声称 flask 未标记/未加培养基 |

32B 在真正送达的 5 个正例上内容表现较好，但没有可靠过滤 3B 的 4 个 false positive。reason 也不应
被当作可靠解释：5 个真阳性中只有 d10 reason 与 GT 对齐。

## 10. 为什么原来 3B 总是 silent

1. **适配遗漏了数据集中已有的状态。** 旧版没有把 `dialog[i]` 和 `task[i]` 传入 3B。这是本次复核
   确认的首要实现错误。
2. **视频证据很短。** 16 帧只有约 0.53 秒，无法稳定判断“步骤刚完成”或动作先后。
3. **重投影进一步删掉时间变化。** 平均只剩 2.24 帧，超过一半的点只剩一帧。
4. **没有 ASR。** 原 LabGate 42-case 中很多触发依赖人工问题或错误描述；Bio 全部是 no speech。
5. **3B 对 prompt 极敏感。** 补入 raw history 后它会过度触发；强调不重复历史后 clean 条件又几乎
   全 NO。它没有学到稳定的时机判别边界。
6. **历史控制标记形成捷径。** raw history 中只保存过去的 `$interrupt$` assistant 话语，保留标记
   明显提高触发率；这不是可靠视觉理解。

所以“全 silent”既不是 prompt 没有调用，也不是 parser 吞掉 YES。它来自适配遗漏与模型/帧输入能力
共同作用。修正遗漏后，问题从“几乎不触发”变成“silent/interrupt 对 prompt 和标签标记剧烈摆动”。

## 11. 为什么 32B 有 current_step 后仍会错

`task[i]` 告诉模型当前步骤是什么，但没有直接表示该动作在画面中是“尚未开始、正在进行、刚完成”。
next-step 标签依赖这个完成状态。局部视频不足时，32B 常把正确进行中的动作解释为未执行或错误动作。
raw history 中的旧警告还会盖过当前 recovery step，导致重复已解决的危险。

此外 3B 的 reason accuracy 只有 7.1%，32B 实际收到的 coarse reason 经常错误。32B 有时能靠
`task[i]` 纠正 reason，有时会被错误 reason 和旧历史带偏。`current_step` 修复了空字段，却不能替代
可靠的 `step_status/completed_step` 状态跟踪。

## 12. 复现命令

生成上下文数据：

```bash
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  LabGate/prepare_egoproactive_bio_contextual.py \
  --dataset /home/gz2522/bio-dataset/EgoProactive-Bio \
  --out-dir /scratch/gz2522/gz2522/VLM4WetExperiment-reference/LabGate/data/bio_contextual_official_20260906
```

运行测试与三个实验入口：

```bash
/scratch/gz2522/gz2522/venvs/bio-proassist-py312/bin/python \
  -m unittest discover -s LabGate -p 'test_*.py' -v

sbatch LabGate/run_bio_gate_contextual.sbatch
sbatch --partition=h100_tandon LabGate/run_bio_expert_contextual.sbatch
sbatch --partition=h100_tandon LabGate/run_bio_contextual_cascade.sbatch
```

已完成作业：gate 开发迭代 `17085408/17086082`（L40S），oracle expert `17086568`（H100），
final cascade `17086574`（H100）；全部完成且 exit code `0:0`。最终测试为 21 项。

## 13. 文件索引

| 内容 | 文件 |
|---|---|
| 修正版 prompt 与输入结构 | [`PROMPTS_BIO_CONTEXTUAL.md`](PROMPTS_BIO_CONTEXTUAL.md) |
| 上下文实验总目录 | [`experiments/egoproactive_bio_contextual_20260906/`](experiments/egoproactive_bio_contextual_20260906/) |
| 25×3 条 final 级联 gate 输入输出 | [`gate_predictions.jsonl`](experiments/egoproactive_bio_contextual_20260906/cascade/gate_predictions.jsonl) |
| 25×3 条 final 级联完整输出 | [`predictions.jsonl`](experiments/egoproactive_bio_contextual_20260906/cascade/predictions.jsonl) |
| final 级联汇总 | [`summary.json`](experiments/egoproactive_bio_contextual_20260906/cascade/summary.json) |
| 14×6 条 oracle-expert 输出 | [`predictions.jsonl`](experiments/egoproactive_bio_contextual_20260906/expert_oracle/predictions.jsonl) |
| oracle-expert 汇总 | [`summary.json`](experiments/egoproactive_bio_contextual_20260906/expert_oracle/summary.json) |
| 32B 人工逐条审核 | [`expert_manual_review.csv`](experiments/egoproactive_bio_contextual_20260906/expert_manual_review.csv) |
| 级联人工逐条审核 | [`cascade_manual_review.csv`](experiments/egoproactive_bio_contextual_20260906/cascade_manual_review.csv) |
| 数据与作业元数据 | [`job_metadata.json`](experiments/egoproactive_bio_contextual_20260906/job_metadata.json) |
| 实际作业源码快照 | [`source/`](experiments/egoproactive_bio_contextual_20260906/source/) |

旧 v1/v2、视觉消融和无 current-step 的 oracle 实验仍保留为失败路径证据，但不再作为正确数据适配的
主结果。实验索引见 [`experiments/README.md`](experiments/README.md)。

## 14. 结果边界

- 数据只有一个剪辑合成视频和 25 个决策点，不能估计跨实验或跨操作者泛化。
- prompt 在同一 25 点上诊断后修改，final 结果是开发集行为，不是独立测试成绩。
- history 是 gold teacher-forced history；真实部署中的 rollout history 可能产生不同误差。
- `task[i]` 含显式 error/recovery 文字，上下文版结果部分依赖强步骤提示。
- 自动语义分数是关键词概念覆盖；人工审核也不是独立领域专家盲审。
- 下一步应在独立数据上训练/验证显式 `step_status`，并把 reason 拆为可校准分类头；在有独立验证集前，
  不宜继续用这 25 点反复调 prompt。
