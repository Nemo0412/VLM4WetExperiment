# EgoProactive-Bio 上下文修正版实验档案

总体解释、模型输入输出和结论见
[`../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。
本目录保存修正“3B 未收到 history、32B current_step 为空”后的原始证据。

## 正式输入

每个决策按同一索引对齐 `video_intervals[i] / task[i] / dialog[i] / answers[i]`。
`task[i]` 作为 `current_step` 传给两个模型；`dialog[i]` 作为真实 chat history 传入。
官方主条件保留高层 query 和最近 4 个 turn。完整累计历史与去除控制标记的历史是消融。
`answers[i]`、reference guidance 和 GT 字段只在生成后评分。

数据文件在 [`data_official/`](data_official/)：25 条 gate records、14 条 oracle-positive expert
records 和 SHA-256 元数据。`data_v1/` 是第一次完整历史 pilot 的旧构造，仅用于追踪迭代。

## 结果

真实级联 `17086574` 的官方 raw + 重投影主条件：

| Gate TP/FP/FN/TN | Precision | Recall | F1 | Expert calls | E2E TYPE acc | E2E semantic complete / 14 positives |
|---|---:|---:|---:|---:|---:|---:|
| 5/4/9/7 | 55.6% | 35.7% | 0.435 | 9/25 | 40.0% | 28.6% |

文件：[`cascade/gate_predictions.jsonl`](cascade/gate_predictions.jsonl)、
[`cascade/predictions.jsonl`](cascade/predictions.jsonl)、
[`cascade/summary.json`](cascade/summary.json)。人工复核见
[`cascade_manual_review.csv`](cascade_manual_review.csv)：5 个已路由真阳性中 4 correct、1 partial；
4 个误触发全部生成错误或不必要指导。

Oracle-triggered 32B 作业 `17086568` 在 14 个正例上给出正确 coarse reason。官方 raw + 重投影的
TYPE accuracy 为 64.3%，自动语义完整率 50.0%，人工为 6 correct / 4 partial / 4 wrong。
文件在 [`expert_oracle/`](expert_oracle/)，人工明细在
[`expert_manual_review.csv`](expert_manual_review.csv)。

## 开发迭代

[`pilot_full_history/`](pilot_full_history/) 是补入完整 history/task、尚未区分旧 user turn 的 pilot。
raw + 重投影 Recall 92.9%，但有 10 个 false positives。

[`development_iteration/`](development_iteration/) 增加“历史不是当前 request”并采用官方四轮历史，
但尚未加入“不重复最近 assistant 指导”的 timing 规则。官方 raw + 重投影 Recall 78.6%，仍有 10 个
false positives。这些输出用于诊断 prompt 敏感性，不作为独立测试成绩。

final cascade 使用 history timing 规则，将主条件 false positives 降到 4，同时 true positives 从
11 降到 5。该变化说明 3B 会在过度触发与过度 silent 之间摆动。

## 文件说明

| 路径 | 内容 |
|---|---|
| `data_official/` | 最终 contextual gate/expert JSONL 与 hash |
| `cascade/` | final 3B routing、conditional 32B 输出、汇总与 run metadata |
| `expert_oracle/` | 32B 解耦输出、汇总与 run metadata |
| `pilot_full_history/` | 第一次上下文 pilot 的 3B 输出 |
| `development_iteration/` | official history + request disambiguation 的 3B 输出 |
| `source/` | 最终作业源代码与 sbatch 快照 |
| `job_metadata.json` | commit、模型 revision、数据 hash、作业/GPU/exit code |

不要跨不同 prompt 迭代拼接预测来构造分数。每个 summary 只使用同一作业内的完整条件。
