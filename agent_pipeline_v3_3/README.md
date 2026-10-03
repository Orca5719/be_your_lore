# Benchmark 3.3 — 基座

本目录先交付 3.3 本体，不包含 3.3A Compact Judge 或 3.3B Deterministic Coverage 的生成行为实验。

当前 Extractor 直接调用 Benchmark 3 使用的 `agent_pipeline_v2.extractor.extract_events`，沿用原提示、窗口、重试和规范化事件协议。新增的 `coverage_accounting` 在生成结束后由程序计算，列出事件、忽略、非事件和仍未覆盖的 span ID。它不更改模型输入、生成上限、检索或 Judge 判决，也不会把缺漏自动视作无事件。3.3B 将在此接口上实验局部恢复。

从项目根目录运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3 validate
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3 run --device cuda
```

`run` 对固定 24 篇故事做一次预热与一次正式采样，逐篇保存并支持同一 `--output` 目录续跑。终端打印 `RESULT_DIR` 与 `SUMMARY`；报告沿用 Benchmark 3 的 profiling 和质量护栏。由于模型再次采样可能与旧运行不同，若质量护栏不一致，退出码为 2 并在报告列出案例，不代表覆盖账本改变了语义。

## 3.3A — Compact Reasoning Judge

3.3A使用Benchmark 3保存的80个事实和检索结果，只重新运行Judge。一次命令顺序测试`structured`和`short-reason`两个候选，分别预热；两个候选都输出证据ID和有限的结构化assessment，后者另外输出一句至多80字的理由。程序沿用“无法同时为真”保护，不让缺少直接证据的明确判断直接进入结果。此实验不改Extractor、Retrieval或3.3本体运行入口。

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3.judge_a_cli run --device cuda
```

输出目录含`judge_a_summary.md/json`、`verdict_changes.csv`、`judge_calls.csv`以及两个候选各自逐篇原子保存的运行文件。报告同时列出Benchmark 3完整版与3.2A极简版，重点比较三分类Macro-F1和`uncertain` Recall。通过`--output <原RESULT_DIR>`可在源文件及两版提示摘要不变时续跑；已完成的两版可以离线重建报告，不加载模型。正式CUDA运行由用户执行；本模块不自行选出新默认Judge。
