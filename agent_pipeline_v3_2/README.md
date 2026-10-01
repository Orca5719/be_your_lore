# Benchmark 3.2 — Lean Generation

本模块按三个独立实验块推进。当前只实现 3.2A Lean Judge；3.2B 与 3.2C 必须等待 3.2A 实测验收。

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
