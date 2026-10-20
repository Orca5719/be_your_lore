# Benchmark 4：Extractor 批处理实验

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

## 4B：连续批处理

4B 仍使用 Benchmark 3 的 Extractor 与修复适配层；只把 `model.generate()` 换成同一 Qwen 模型的逐步 greedy forward。新请求独立 prefill；活跃请求共用 decode 步。一条请求遇到 EOS 或输出上限后立刻释放槽位，下一请求随后加入。KV cache 只保留活跃请求，重新组批时按各请求的有效长度对齐。容量 1、2、4、8 分别预热后正式处理相同的24篇故事。

先检查冻结输入与 4A 结果：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 validate-continuous
```

正式运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 run-continuous --device cuda --capacities 1 2 4 8
```

默认对照 `agent_pipeline_v4/reports/benchmark_4A_20261004T050254Z_1774b9`；如使用别的4A目录，传 `--static-result <目录>`。输出开头打印 `RESULT_DIR=`，结束打印 `SUMMARY=`。中断后用 `--output <原RESULT_DIR>` 续跑；续跑档位的 Wall 时间标空，不用于正式速度对照。离线重建：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4 summary-continuous --result-dir <RESULT_DIR>
```

报告包含逐请求trace、逐forward步数据、槽位利用率、排队与首token时间、KV cache整理开销、无效生成步数及对应4A档位对照。4A只保存了解码文本和输出token数，没有原始token ID；逐案差异只能比对解码文本、计数、事件和质量，不声称已证明token序列逐位相同。4A的batch 4/8曾有一篇partial；4B保留这种质量变化并单列，不把它改成性能收益。若容量 8 显存不足，记录OOM，不自动降档。
