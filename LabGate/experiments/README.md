# EgoProactive-Bio 实验文件索引

请先阅读 [`../EGOPROACTIVE_BIO_EVALUATION_REPORT.md`](../EGOPROACTIVE_BIO_EVALUATION_REPORT.md)。
该文件统一描述数据、模型输入输出、正式流程、指标、全部结果与分析。

本目录下各子目录只承担原始证据留档：

| 目录 | 定位 | 是否作为最终结论 |
|---|---|---|
| [`egoproactive_bio_20260905/`](egoproactive_bio_20260905/README.md) | v1：专用 silent/interrupt prompt 与 gold/rollout history；用于发现历史标签偏置 | 否 |
| [`egoproactive_bio_v2_20260905/`](egoproactive_bio_v2_20260905/README.md) | v2：JSON reason/observation/state；用于记录结构化输出退化 | 否 |
| [`egoproactive_bio_native_20260905/`](egoproactive_bio_native_20260905/README.md) | 未传 history/task 的原生 LabGate 视觉消融 | 否，保留作 all-silent 诊断 |
| [`egoproactive_bio_decoupled_20260905/`](egoproactive_bio_decoupled_20260905/README.md) | 未传 current step 的 3B/32B 解耦消融 | 否，保留作缺失状态对照 |
| [`egoproactive_bio_contextual_20260906/`](egoproactive_bio_contextual_20260906/README.md) | 修正 history/current_step 后的 gate、oracle expert、真实级联和人工审核 | **是，当前主证据** |

不要跨目录拼接单次预测来构造新分数。同一条件在 H200/L40S 上存在一个边界样本差异；
每个汇总只对应自己目录中的完整运行和硬件环境。
