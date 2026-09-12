# EgoProactive-Bio 上下文版模型输入与 Prompt

本文记录修正后的实际模型输入。对应代码为
[`models.py`](models.py)、[`prompts.py`](prompts.py)、
[`eval_egoproactive_bio_gate_contextual.py`](eval_egoproactive_bio_gate_contextual.py)、
[`eval_egoproactive_bio_expert_contextual.py`](eval_egoproactive_bio_expert_contextual.py) 和
[`eval_egoproactive_bio_contextual_cascade.py`](eval_egoproactive_bio_contextual_cascade.py)。

## 1. 每轮上下文如何对齐

对第 `i` 个决策点，数据直接按相同下标读取：

```text
video_intervals[i] -> 当前决策时间区间
task[i]            -> current_step
dialog[i]          -> 当前决策之前的累计对话 history
answers[i]         -> 只用于 GT 与评分
```

`dialog[i]` 是历史，不包含当前轮模型输出。`task[i]` 同时提供给 3B 和 32B。
`answers[i]`、参考指导、`gt_type` 和 `gt_reason` 不进入真实级联的模型输入。

历史以真实 chat turns 传入 Qwen，而不是拼成一段审计文本：

```python
messages = [
    *history,  # 每项是 {"role": "user"|"assistant", "content": ...}
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": current_prompt},
        ],
    },
]
```

正式 WearableAI 对齐方式保留第一条高层 query，并从 query 之后只取最近 4 个 turn。测试包含：

- `history_official_raw`：高层 query + 最近 4 个历史 turn，保留 `$interrupt$`。
- `history_official_clean`：相同官方截断历史，只移除 `$interrupt$/$silent$` 控制标记。
- `history_full_raw/full_clean`：传入未截断的完整累计历史，用来检验长历史的影响。

official raw 是最接近现有 WearableAI pipeline 的主条件；clean 和 full 是消融条件。

## 2. d03 的实际对齐示例

```json
{
  "id": "d03",
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

重投影条件会从这 16 帧中进一步筛选；full-16 条件直接使用全部 16 帧。

## 3. 3B Judger 输入

3B 收到：历史 chat turns、当前视频帧，以及当前 user turn 中的下列 prompt。

```text
You are a fast lab-monitor gate. Decide whether the LARGE assistant VLM must be
called on this moment.

Any conversation turns before the current video are HISTORY from earlier
decisions. Never treat an earlier user turn as a new request now. The CURRENT
TASK / STEP is supplied state context, not user speech or a request. Only the
current USER SPEECH (ASR) field below may trigger reason=user_query; when it says
(no user speech), reason=user_query is forbidden.

PROTOCOL:
{static_cell_passaging_protocol}

CURRENT TASK / STEP (provided context):
{task[i]}

USER SPEECH (ASR):
{asr_text_or_(no user speech)}

Call YES if ANY of these is true:
  1. Safety issue ...
  2. User asked a question or requested help ...
  3. Visible actions conflict with the protocol ...
  4. A protocol step has visibly just completed and a timely next-step reminder
     would help; use reason=next_step.
History controls timing: if the current action is the correction or next action
already requested by the latest assistant message, reply NO when it is being done
correctly. Do not repeat an earlier instruction. Use next_step only after the current
step has visibly completed and the needed next action has not already been given.
A provided recovery step being performed correctly is NO unless a new problem exists.
When unsure about a protocol conflict, prefer YES so the large model can check.
Otherwise NO. Routine correct work with silence -> NO.

Reply with EXACTLY one line:
YES. reason=<safety|action_error|user_query|next_step>
or
NO.
```

输出由锚定在首 token 的解析器读取。只有以 `YES` 开头才触发；正文中偶然出现的
`yes` 不会触发。以 `NO` 开头或非法输出都不调用 32B。

## 4. 32B Expert 输入

真实级联中只有 3B 触发后才构造这一轮输入。32B 收到与 3B 完全相同的历史、视频帧和
`task[i]`，并额外收到 3B 的预测 reason。触发对象实际为：

```json
{
  "reason_type": "<3B predicted safety|action_error|user_query|next_step|unknown>",
  "observation": "",
  "current_step": "<the exact same task[i] sent to 3B>"
}
```

其中 `observation` 为空是因为当前 3B 协议只输出判断和 reason，没有生成可靠的视觉观察句。
`current_step` 不再为空。32B prompt 的关键结构是：

```text
You are a wet-lab copilot watching first-person video plus the protocol.

PROTOCOL:
{static_cell_passaging_protocol}

CURRENT TASK / STEP (provided context):
{task[i]}

USER SPEECH (ASR):
{asr_text_or_(no user speech)}

SMALL VLM TRIGGER (unverified; check against the current visual evidence):
{"reason_type": ..., "observation": "", "current_step": task[i]}
Do not assume the proposed reason is correct or repeat a resolved historical warning.

Produce exactly one decision:
  SAFETY       — hazard / unsafe handling. Warn immediately.
  ACTION_ERROR — what the person is doing conflicts with the protocol.
  ASSISTANT    — user asked something or the trigger reason is next_step; give timely guidance.
  NONE         — nothing to report.

Reply in EXACTLY this format (two lines):
TYPE: <SAFETY|ACTION_ERROR|ASSISTANT|NONE>
MSG: <one concise instruction, or empty for NONE>
```

Oracle-expert 解耦实验中的 `reason_type` 使用 GT coarse reason，明确作为能力上界；真实级联只转发
3B 实际预测，不使用 GT。

## 5. 落盘审计字段

每条 3B 输出保存 `history_sent`、`current_step_sent`、完整 `prompt`、`raw`、解析后的
`fired/reason`、帧数和帧索引。每条 32B 输出还保存完整 expert prompt、`expert_raw`、
解析后的 `pred_type/message`。因此可以逐条确认模型看到了什么，评分字段不会被误认为模型输入。

`task[i]` 中有 `error`、`recovery` 和 “Do not ...” 等显式语义。这是用户要求保留的数据集输入，
也意味着上下文版分数带有强步骤提示，必须与不含 `task/history` 的视觉消融分开报告。
