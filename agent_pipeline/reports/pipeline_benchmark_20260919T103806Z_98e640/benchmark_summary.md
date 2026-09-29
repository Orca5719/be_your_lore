# Agent Pipeline Benchmark 表（2026-09-19）

结果目录：`pipeline_benchmark_20260919T103806Z_98e640`

> 当前数据集是4段故事、5条gold事实的草案，只能作为provisional baseline。处理状态ok表示程序流程完成，不等于语义正确。

## 系统与运行

| Metric | Result |
|---|---:|
| LLM | Qwen/Qwen3-4B-Instruct-2507 |
| Parameters | 约4B |
| LLM weights / compute | NF4 4-bit / BF16 compute |
| Retriever | BAAI/bge-small-zh-v1.5，FP32，Top-K=5 |
| Device | CUDA |
| Cases | 4 |
| Gold facts | 5 |
| Processing status | 3 ok / 1 partial |
| Processing completion rate | 75% |
| Model load | 15.68 s |
| Encoder + index load | 0.50 s |
| Peak CUDA allocated | 4.43 GiB |
| Peak CUDA reserved | 6.43 GiB |
| Separate model VRAM | 未单独记录 |

## Quality

| Metric | Numerator / Denominator | Result |
|---|---:|---:|
| Recall_Extraction | 4 / 5 | 80.0% |
| Recall@5_Retrieval | 3 / 4 | 75.0% |
| Accuracy_Judge | 4 / 4 eligible | 100.0% |
| EndToEnd conflict Precision | TP=0, FP=0 | N/A |
| EndToEnd conflict Recall | TP=0, FN=1 | 0.0% |
| EndToEnd conflict F1 | TP=0, FP=0, FN=1 | 0.0% |

Judge Accuracy只统计已提取、完成检索且满足证据要求的eligible事实。SL-P03的矛盾事实在提取阶段漏掉，因此没有进入Judge分母；100%不能解释为端到端100%。

## Latency and generation

| Metric | Result |
|---|---:|
| End-to-end median | 160.63 s / story |
| End-to-end mean | 179.46 s / story |
| End-to-end P95 | 264.64 s / story |
| Qwen generation attempts | 41 |
| Median TTFT | 832.45 ms |
| Mean TTFT | 1107.21 ms |
| Weighted decode speed | 11.53 tok/s |
| Average input tokens / attempt | 1267.6 |
| Median input tokens / attempt | 1067 |
| Total decoded tokens | 7749 |

以上生成统计不含warmup，按understanding、引用修复、filtering、judge与story_check的实际attempt汇总；端到端时延包含非LLM阶段和所有重试。

## Per-case

| Case | Status | Latency | Gold coverage | Result |
|---|---|---:|---:|---|
| SL-P01 | ok | 164.28 s | 无gold | 普通故事处理完成；E1被误保留为mechanism，但未产生矛盾误报 |
| SL-P02 | partial | 264.64 s | 3/3 | 三条双心脏事实均正确提取、Top-5召回并判一致；“鞋带有些松”主体被错写为“他”，留下真实语义失败 |
| SL-P03 | ok | 131.94 s | 0/1 | 漏掉“亚巴顿寄宿在雷的右侧心脏”，因此没有召回和判断，错失唯一矛盾gold |
| SL-P04 | ok | 156.98 s | 1/1 | 时间暂停能力被提取并最终判不确定，符合当前gold |

## Main limitation exposed by this run

SL-P03的关键片段S3只作为另一事件的context_ids出现。当前coverage把context引用也视为“已覆盖”，所以没有触发补提，最终工程状态仍为ok。这说明当前最大瓶颈已经回到第一阶段语义提取：流程稳定性提高了，但`ok`仍不能保证重要事实都成为独立事件。
