# Agent Pipeline Quality Benchmark — provisional

> 2026-09-19：首轮运行暴露窗口上下文膨胀，随后已在understanding-v7/filtering-v7修复。第二轮`pipeline_benchmark_20260919T091538Z_cc2a2b`的核心5条草案事实全部走通，但4例仍因路由和行级失败隔离问题均为partial。当前understanding-v8加入显式非事件片段，筛选与段内检查改为坏项局部重试；两个旧目录都只作为历史基线，必须新建目录重跑当前版本。

当前使用冻结的4段Story-Level pilot草案，共5条事实、48个索引片段。金标状态是annotation_draft_not_final_benchmark，结果只作初步基线。旧benchmark文件不改。

指标：Recall_Extraction以全部gold事实为分母；Recall@K_Retrieval只统计有核心证据要求的gold；Accuracy_Judge只统计已语义匹配且所需证据已召回的eligible gold；EndToEnd P/R/F1以矛盾事实为正类，漏提/漏召回/失败为FN，错误矛盾和段内误报为FP。Latency_EndToEnd包含理解、筛选、检索、设定判断、筛选后段内互检和报告；模型/BGE加载单列，预热不计案例时延。

运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_benchmark.py" run --device cuda --top-k 5 --warmup 1
```

运行完成后必须人工审核review.json中的gold到event覆盖关系并设status=reviewed，再执行score。允许一个事件覆盖多个原子gold事实，禁止仅按原文重叠或模型标签自动配对。

Step 53修复段内互检绕过filtering的问题：纯日常事件应在筛选阶段排除，不再形成二次方比较。若模型错误保留大量日常事件，延迟和误报仍由benchmark真实记录。

当前结构规则：`unresolved_actor`不会阻塞模型明确判定为routine/process_detail的忽略；若模型要保留该事件，则转为review。合法连接/依附片段通过`non_event_span_ids`完成覆盖审计，不强造事件。筛选中单条非法决定、段内检查中单条非法pair只重试该项；同批合法结果不会被丢弃。reason_code被无歧义复制进decision时按固定映射规范化；非矛盾段内结果不保留无关引用，矛盾引用仍逐字严格校验；coverage补提多输出的context-only副本只审计。下一轮仍须重新人工审核新目录的review.json，旧映射不能直接复制成当前成绩。

## 冻结基线

`pipeline_benchmark_20260919T103806Z_98e640` 已作为 Agent Pipeline Quality Benchmark v1 冻结在 `benchmark_freezes/agent_pipeline_v1/`。快照包含当时的运行代码、提示、schema、依赖锁、4段草案数据、示例 lore、实际使用的向量索引和完整结果目录；不包含可从 Hugging Face 重新取得的模型权重。

该成绩保留原有的 `annotation_draft_not_final_benchmark` 与 `scored_provisional` 标记，不能当作正式泛化成绩。只读校验命令：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline.verify_freeze
```
