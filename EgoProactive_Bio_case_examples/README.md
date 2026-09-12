# EgoProactive-Bio：一个 E2E 成功案例和一个失败案例

本目录从最终的 `official_raw_reprojection` 条件中各选一个案例，目的是让读者能直接核对：

1. 数据集标注和 `current_step` 是什么；
2. 3B Judger 看到了什么、输出了什么；
3. 3B 触发后，32B Expert 输出了什么；
4. 重投影究竟给模型保留了多少视觉信息。

两个案例都来自同一个源视频 `bio_cp_oop_full_001.mp4`。这里的 `official_raw` 表示输入为初始
任务 query 加最近 4 条历史消息，并保留 `$interrupt$`；`reprojection` 表示先从决策区间中点取
16 个连续帧，再用 `tau=0.12` 的重投影门过滤相似帧。

## 文件怎么看

每个案例目录包含：

- `context_clip.mp4`：标注的约 8 秒区间，供人检查动作上下文；为了便于查看，导出为
  540×720、15 FPS。这不是模型直接输入。
- `sampled_16_frames.jpg`：模型过滤前取得的 16 帧，标有局部下标、源视频帧号和时间。
- `model_kept_*.png`：评估日志记录的重投影保留帧，即实际送入模型的视觉内容。单帧会在
  Qwen 包装器中复制为两帧以满足视频输入格式，但不会增加时间信息。
- `history.json`、`judger_prompt.txt`、`judger_output.txt`：3B 的历史、当前 prompt 和原始输出。
- `expert_prompt.txt`、`expert_output.txt`：真实 E2E 中触发 32B 时的 prompt 和输出。失败案例
  没有这两个文件，因为 32B 当时没有被调用。
- `oracle_expert_prompt.txt`、`oracle_expert_output.txt`：解耦诊断中强制调用 32B 的结果，不能
  当成真实 E2E 输出。
- `case.json`：上述信息的结构化快照，包括 ground truth、延迟、帧索引和解析后的输出。

## 成功案例：d10

目录：[success_d10](success_d10/)

数据集标注：

```text
current_step: Step 7 error — Do not leave the flask dry after removing culture medium
ground-truth trigger: YES
ground-truth reason/type: action_error / action_error
reference: Do not leave the flask dry and exposed. Add PBS immediately to rinse the cells.
```

3B 输出：

```text
YES. reason=action_error
```

因此系统调用32B。32B输出：

```text
TYPE: ACTION_ERROR
MSG: The flask appears to have been left dry after removing the culture medium.
Please promptly rinse the cells with PBS and discard the PBS to maintain cell viability.
```

这是一个 E2E 成功案例：3B 的触发和理由都正确，32B 的类型正确，指导也覆盖了关键动作
“立即加入 PBS 冲洗”。

需要注意，16 个原始采样帧是源帧 3839–3854，只覆盖 127.967–128.467 秒；重投影最终仅保留
局部帧 0，即源帧 3839。成功不能完全归因于视频识别，因为 `current_step` 本身包含 `error` 和
“Do not leave the flask dry”，raw history 还保留了 `$interrupt$` 控制标记。这是包含标注状态的
contextual pipeline 测试，不是纯视觉测试。

## 失败案例：d14

目录：[failure_d14](failure_d14/)

数据集标注：

```text
current_step: Step 10 error — Do not shake the flask vigorously after adding trypsin
ground-truth trigger: YES
ground-truth reason/type: action_error / action_error
reference: Avoid vigorous shaking. Gently rock or tap the flask so the trypsin is distributed evenly.
```

3B 实际输出：

```text
NO.
```

因此真实 E2E 流程到这里结束，32B 没有收到请求，也没有真实 E2E 指导文本。这是一个 false
negative，失败发生在 3B 路由阶段。

为定位问题，解耦实验使用正确的 `action_error` 理由强制调用了32B。32B输出：

```text
TYPE: ACTION_ERROR
MSG: The person appears to be tilting the flask excessively, which could lead to vigorous shaking.
Please handle the flask gently to avoid disturbing the cells unnecessarily.
```

这个强制调用结果的类型和语义基本正确，说明该案例中32B有能力给出合理指导；E2E失败主要由
3B漏触发造成。

d14 的16帧是源帧7488–7503，只覆盖249.600–250.100秒，重投影同样只保留第一帧7488。
“剧烈摇晃”是时间动作，单张静态图很难证明动作幅度。8秒 `context_clip.mp4` 对人更容易判断，
但模型没有看到整段8秒视频。这个案例直接展示了当前中点0.53秒采样和重投影过滤可能删除动作
证据的问题。另一方面，`current_step` 已经明确写有 `error`，3B仍输出 `NO`，也说明其输出不只
受视觉证据限制，还存在 prompt 遵循不稳定的问题。

## 两个案例放在一起说明什么

两个案例的重投影输入都只有一帧，而且 `current_step` 都带有明确的 `error` 描述，但3B一次输出
YES、一次输出NO。这符合完整实验观察到的现象：当前 gate 对 prompt、历史和有限视觉证据较敏感，
路由行为不稳定。d14 的 oracle 32B 结果进一步说明，至少对这个例子，主要瓶颈是3B是否触发，
而不是32B是否能组织指导语句。

## 重新生成

在项目根目录运行：

```bash
python EgoProactive_Bio_case_examples/build_examples.py
```

脚本从最终实验产物读取输入输出，再从原数据集视频生成片段和帧图。它不会重新运行模型。
