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

## 4.1 Debug：四组受控首窗口实验

4.1质量表显示多请求输出发生语义变化，但表格本身不能证明KV cache串话。针对`SL-010`第一提取窗口（历史CB1首轮成功、CB4首轮和重试均报事件字段错误），复用**未改动的**`QwenContinuousEngine`运行：

- A：目标独自运行。
- B：目标与历史首次decode同批的`SL-P04`、`SL-009`、`SL-P03`，目标处于槽位3。
- C：目标与`SL-011`、`SL-012`、`SL-013`，目标仍处于槽位3。
- D：与B相同邻居，目标移到槽位0。

四组都重建相同的目标首窗口prompt，邻居也只用各自首窗口；一次模型加载，不重跑24篇，不触发提取器的格式重试或覆盖恢复。每步记录目标实际token、top-1/top-2 token及logit、位置、batch顺序和cache重组次数。原24篇运行时，邻居可能在目标入场前已decode若干步，因此这个受控实验不等于逐步复刻原调度。

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 validate-debug
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 run-debug --device cuda
```

运行打印`RESULT_DIR`及`SUMMARY`。同一目录可用`--output "<RESULT_DIR>"`断点续跑，或离线重建摘要：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_1 summary-debug --result-dir "<RESULT_DIR>"
```

`debug_summary.md/json`列出A/B/C/D首次分歧；`scenario_*.json`保存逐token原始证据。邻居或槽位变化造成的差异只定位可疑边界，不能单凭差异断言KV泄漏；需看首分歧的top-2间隔、logit幅度、position/cache位置，并考虑不同batch宽度的数值路径。
