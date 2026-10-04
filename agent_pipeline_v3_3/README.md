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

### 3.3A追加对照：assessment与理由长度

首轮结构化版和80字理由版都未达到完整版Judge的质量，因此追加三组独立实验：

- `assumptions-list`：恢复旧版assessment的缺失前提文字列表，不写理由。
- `rationale-120`：同一assessment，增加至多120字理由。
- `rationale-240`：提示与上一版除长度数字外相同，只将理由上限增至240字。

三组都保留原判决保护，仍从冻结的Benchmark 3事实/证据输入，避免案例特化。程序自动并排报告完整版、3.2A极简版、首轮两版与本轮三版，共七列。运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3.judge_a2_cli run --device cuda
```

打印`RESULT_DIR`和`SUMMARY`；各候选逐篇保存，可用`--output <原RESULT_DIR>`续跑。本轮仍不自动选默认Judge，正式生成由用户执行。

### 3.3A输入对齐修正（2026-10-04）

复核追加实验时发现：旧Full Judge输入为`event`对象，早期A/A2候选却收到`fact`对象；两者虽然来自同一条提取事实，但字段和语义提示不同。因此`benchmark_3_3A_20261003T154610Z_7305ef`及`benchmark_3_3A2_20261003T160701Z_ac16fe`只能视为**混合输入与输出差异的历史观察**，不能单独归因于输出长度。

现在五个A候选使用Full Judge原有的输入构造函数，保持用户消息逐字相同，仅系统提示和输出协议不同。已对固定资料逐事实验证输入一致。必须生成新结果目录，旧manifest会拒绝续跑。为同时得到五个候选的可比结果，可在项目根目录依次运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3.judge_a_cli run --device cuda --output ".\agent_pipeline_v3_3\reports\benchmark_3_3A_aligned"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3.judge_a2_cli run --device cuda --a1-result ".\agent_pipeline_v3_3\reports\benchmark_3_3A_aligned" --output ".\agent_pipeline_v3_3\reports\benchmark_3_3A2_aligned"
```

第二条命令会读取第一条的新报告，生成七列对照；两条都可在各自的`--output`目录安全续跑。正式CUDA运行仍交给用户。

## 3.3B — Deterministic Coverage 与局部补提

3.3B只重新运行Extractor。模型、原版Extractor提示、窗口划分、事件字段和1536-token生成上限保持不变；从3.3基座的24篇、80事实结果读取对照。程序先严格验证模型给出的事件与处置，独立计算未覆盖span。若首轮JSON有效但漏掉span，已经验证的事件和处置先保留，仅把遗漏ID和邻近上下文交给模型局部补提；局部有效但仍有缺口时最多继续一轮。若局部补提格式或字段错误、或者两轮后仍有缺口，再做一次完整窗口兜底。兜底完整通过时以其结果为准；失败时保留先前有效内容并明确列出未覆盖ID，绝不自动写进忽略或非事件。首轮格式或字段本身无效也允许一次原窗口重试。

在项目根目录运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_3.extractor_b_cli run --device cuda
```

终端打印`RESULT_DIR`和`SUMMARY`。结果包含`extractor_b_summary.md/json`、逐篇`extractor_runs.jsonl`和`event_review.json`。每篇完成后原子保存；中断后用相同的`--output <RESULT_DIR>`续跑。也可以用`summary --result-dir <RESULT_DIR>`离线重建报告，不加载模型。manifest固定资料、基座、原提示、代码和模型版本，不允许混用不同实验。

报告比较调用数、输入/输出token、Prefill/Decode、总耗时、显存、被保留的首轮不完整输出、后来被兜底替换的输出、局部补提、整窗兜底和真正无效的生成。新事件若与原人工审核的事件不完全相同，会列入`event_review.json`待复核；此时质量数字只是已有标签的下界，不能宣称正式Recall已提升或下降。3.3B不运行Judge，A阶段的候选也不会混入。

`benchmark_3_3B_20261004T031242Z_b71052`是旧恢复策略的失败记录：24篇中2篇为partial，8个span未覆盖，提取Gold命中由71/72降为66/72。新兜底行为改变了manifest；必须使用新的结果目录重跑，不可在该旧目录续跑。
