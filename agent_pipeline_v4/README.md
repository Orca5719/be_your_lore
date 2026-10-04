# Benchmark 4A：Extractor 固定批处理

本模块只改变 Benchmark 3 Extractor 的生成执行方式。窗口划分、提示、JSON 校验、格式重试、覆盖缺口恢复及人工审核基准仍使用 Benchmark 3 的实现和数据。Retriever、Judge、Report 未在本实验重跑；表中质量是 **Extractor-only** 成绩。

在 `C:\Users\Xhang\Desktop\project\worldcheck` 中运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 validate
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 run-static --device cuda --batch-sizes 1 2 4 8
```

命令先用一篇故事比对 Benchmark 3 单条生成与 batch 1，再分别预热并处理 24 篇正式故事。`RESULT_DIR=` 是结果目录；四档完成后查看 `SUMMARY=` 指向的 Markdown 表，同时保留 JSON、CSV、逐故事结果和每次 `model.generate()` 的 trace。若中断，可加 `--output <原RESULT_DIR>` 续跑。续跑档位的 Wall 时间不再用于正式横向对比；想重测性能请用新目录。

不同 batch 可能改变模型输出。只有与 Benchmark 3 已审核事件的语义字段完全一致时才沿用原标签；新增或变化事件写在各档位的 `pending_event_review.json`，正式 Precision／Recall 显示为“待复核”，不能把已复用事件的下界当成最终成绩。

离线重建表格：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 summary --result-dir <RESULT_DIR>
```

4B 连续批处理将在 4A 实测与复核后实施。
