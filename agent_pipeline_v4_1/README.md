# Benchmark 4.1：连续批处理提取质量

本模块只评测 Benchmark 4B 已保存的 CB2／CB4／CB8 提取结果。提取器、提示、batch 调度和模型输出不改动。端到端实验直接使用 Benchmark 3 的 hybrid RRF（无 metadata filter）、Judge v2.1（batch 8）和 Report；不会重新提取故事。

`review.json` 是逐事件审核账本，锁定数据集及三个提取文件 SHA-256。每个事件都记录原样内容签名、标签、对应 Gold ID 和审核说明；脚本拒绝缺失、旧签名或非法 Gold 映射。`build_review.py` 保存生成账本的审计过程，评测运行不需要调用它。原 2.2 标签只有在事件语义字段完全一致时才用于初始化；本轮逐案核对了引用原文与 Gold，修正了内容变更后仍沿用 E1 等 ID、宿主事实被“拥有两颗心脏”替代、丢失关键时间限定等误标。审核来源是助手，不等同作者确认。

在仓库根目录执行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 validate
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 score-extraction
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 run-downstream --device cuda
```

最后一条命令会打印 `RESULT_DIR`，逐故事原子保存至该目录，可用相同目录断点续跑：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 run-downstream --device cuda --output "<RESULT_DIR>"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 summary --result-dir "<RESULT_DIR>"
```

自动生成 `quality_summary.md/json` 与 `event_audit.csv`。Extraction Recall 的分母为72个 Gold 事实；Precision、Hallucination、Overselect 的分母为该容量输出事件数。Hallucination 指事件陈述缺少其引用的故事原文依据，不评价该陈述在设定库中是否为真；Overselect 指有原文依据但不值得核对的普通事件；Duplicate 单列，三种错误标签互斥。End-to-End F1 的正类是 Gold 中的明确矛盾；没有对应 Gold 的矛盾判断也计 FP。CB4 的 `SL-010` 提取 partial 如实保留并计入漏提，不修复模型输出。

本轮是24篇助手编写的草案故事；4B 各容量只采样一次。后续真实故事和作者审核可能改变绝对分数。
