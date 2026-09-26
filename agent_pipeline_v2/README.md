# Agent Pipeline v2

V2与冻结的v1并列存在，只修改两个实验变量：

1. `extractor.py`用一个LLM阶段直接输出值得核对的事件，并将其余原文审计为普通内容或非事件片段。
2. `judge.py`把独立fact+lore Prompt组成真正的GPU tensor batch，支持batch=1/2/4/8。

流程：

```text
story -> one-stage extractor -> BGE Top-5 -> batched Judge -> deterministic report
```

## 快速验证

此命令不加载Qwen：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" validate
```

## 完整实验

历史 batch=1/2/4/8 结果仅保存在既有报告中，新运行不再重复生成比较表。

正式新运行固定使用Judge batch=8和3次独立重复：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" run-all --device cuda --judge-batch-size 8 --repeats 3 --top-k 5
```

`run-all`先在独立进程运行24篇故事，再为batch=8启动3个全新的Judge进程。24篇故事包含72条draft gold fact，三类判断各24条，当前仍需人工审核语义映射；因此首次运行的故事质量状态是`pending_human_review`。Judge fixture会直接得到三次运行的Accuracy和性能分布表。

结束时会分别打印`STORY_TRACK`、`JUDGE_TRACK`和总`RUN_STATUS`，并保存`run_status.json`。某一轨失败不会阻止另一轨完成；Story partial 会保留为质量结果，不会被误报成进程崩溃。

只运行Judge batching：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" run-judge --device cuda --batch-sizes 8 --repeats 3
```

完成审核和评分文件后生成统一报告：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" summary --result-dir "结果目录"
```

人工审核`story_review.json`、将`status`改为`reviewed`后：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" score-stories --result-dir "结果目录"
```

当前24篇故事、72条gold和48条Judge fixture仍是draft/provisional数据；故事标注尚未完成用户语义审核，不能代表模型泛化能力。
