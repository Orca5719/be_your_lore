# Agent Pipeline v2 Benchmark

## 实验问题

V2回答两个问题：一次“理解并筛选”的结构化提取能否减少信息损失和LLM调用；真正的GPU batching如何改变Judge的显存、总耗时、吞吐和准确率。

V1保持冻结。V2不更换Qwen模型、BGE模型、lore或索引，因此差异主要来自pipeline边界和Judge执行形状。

## 数据

- Story track：`story_benchmark_pilot_4_v1.json`，4篇故事、5条草案gold，用于v1/v2端到端比较。结果必须人工审核gold到event映射。
- Judge track：`agent_pipeline_v2_judge_48.json`，从160条Fact-Level题库稳定抽取，一致/矛盾/不确定各16条。固定fact、story context和最多5条lore；四种batch使用完全相同的48条输入。

两个数据轨均标记为`annotation_draft_not_final_benchmark`。

## Judge计时边界

`total_judge_seconds`包含chat template/tokenize、左侧padding、`model.generate()`、decode、JSON解析、严格验证及失败项重试。不包含模型加载、fixture构建和BGE retrieval。`generation_seconds`另行记录同步后的GPU生成时间。

每个batch size在干净子进程中加载模型，先用一个完整batch预热，再重置CUDA peak counters并处理48条计分输入。固定顺序，不按长度分桶。默认一次计分，`--repeats`可以增加重复次数；多次结果保留原始run并以中位数汇总。

## 指标

- `peak_allocated_gib`：计分阶段PyTorch实际峰值分配，包括模型。
- `peak_reserved_gib`：计分阶段CUDA allocator峰值保留。
- `incremental_peak_allocated_gib`：peak allocated减模型加载后allocated。
- `facts_per_second = 48 / total_judge_seconds`。
- `judge_accuracy = 正确结论 / 48`；解析失败、OOM和其他执行失败均在分母中。
- 另记逐类Accuracy、混淆矩阵、retry rate、parse failure rate、useful/padded input tokens、padding ratio、generated tokens及主/重试batch数。

OOM明确记录请求的batch size，不自动降档。

## 一步提取覆盖协议

每个target span必须且只能进入：`events`、`ignored_spans`、`non_event_span_ids`之一。进入`ignored_spans`的内容仍被审计，只是不送入Retrieval/Judge。程序只规范化能唯一确定的wire错误：单值数组、最近明确前文中的主体依据、纯context重复事件/处置，以及重复的忽略/非事件处置。普通格式错误最多整体重试一次；若其余行均有效而仅漏了target，则保留有效行并只补提缺失target一次。未知主体、未知编号、target/context混合source和语义冲突仍保持失败。

## 命令

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" validate
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" run-all --device cuda --batch-sizes 1 2 4 8 --repeats 1 --story-judge-batch-size 4
```

运行目录保存在`agent_pipeline_v2/reports/`，包含故事原始结果、人工review模板、每个batch/repeat的原始结果、batch聚合结果、Markdown对比表及源码/数据哈希。`run-all`另存`run_status.json`并在终端分别打印故事轨、Judge轨与总状态。
