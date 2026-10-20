# Benchmark 3.2 — Lean Generation

本模块包含三个独立实验块：3.2A Lean Judge、3.2B Extractor Failure Audit、3.2C Lean Extractor。历史结果保留在各自的报告目录。

3.2A 冻结 Benchmark 3 已保存的 Extraction 与 Retrieval，只重新运行 Judge。模型输出仅包含 `verdict` 和 `evidence_ids`，不生成理由或 assessment。

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_2 validate
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_2 run-judge --device cuda --judge-batch-size 8
```

## 3.2B Extractor Failure Audit

本步骤只读取 Benchmark 3 已保存的 trace，不加载模型：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_2 audit-extractor `
  --source-result ".\agent_pipeline_v3\reports\benchmark_3_20260927T142334Z_3ceb30"
```

输出包括失败分类、逐调用成本、成本闭合以及按实际损失排序的3.2C schema假设。

## 3.2C Lean Extractor

模型输出精简为事件核心字段与两个ID账本；适配层恢复旧规范结构。正式运行固定CUDA，并只复用完全相同事件的旧审核决定；变化事件标记pending。

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_2 run-extractor --device cuda
```

审核变化事件后，用纯离线命令重算质量报告，不加载模型：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_2 score-extractor `
  --result-dir ".\agent_pipeline_v3_2\reports\benchmark_3_2C_20261002T161632Z_0883f1" `
  --review-file ".\agent_pipeline_v3_2\reviews\benchmark_3_2C_20261002T161632Z_0883f1.json"
```

审核文件必须精确覆盖全部pending事件，并绑定具体运行的`extractor_runs.jsonl`摘要。正式报告写在结果目录的`reviewed`子目录。召回率按`故事ID + gold ID`计数，避免24篇故事重复使用G1/G2/G3时错误去重。

这轮24篇的审核结果为58/72条gold命中，低于Benchmark 3的71/72；普通动作误提取14条。3.2C质量门槛失败，因此Lean Extractor不是新默认。提示词v2和额外恢复调用同时改变了实验条件，当前速度对照反映改后系统整体。
