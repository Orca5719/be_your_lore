# Agent Pipeline Benchmark 2.1 总报告

审核来源：**assistant-reviewed**，未冒充用户人工复核。24 篇运行全部完成。

## 2.1A — Oracle Judge

| 指标 | Judge v1 | Judge v2.1 |
|---|---:|---:|
| Accuracy | 65.28% | 69.44% |
| Macro-F1 | 52.23% | 68.45% |
| 矛盾误报 | 16 | 4 |
| 矛盾漏报 | 5 | 11 |
| 无依据推断 | 6 | 1 |
| 解析失败 | 4 | 0 |
| Judge 总耗时 | 340.37 s | 407.17 s |
| 峰值显存 allocated | 3.70 GiB | 4.36 GiB |

2.1 的“无法同时为真”规则有效压低了误报，但代价是更保守、漏掉更多真实矛盾。

## 2.1B — Retrieval

| 方法 | Metadata filter | Complete Recall@5 | Evidence Recall@5 | MRR@5 |
|---|---|---:|---:|---:|
| Dense | 关 | 69.44% | 69.57% | 66.09% |
| BM25 | 关 | 84.72% | 86.96% | 77.78% |
| Hybrid RRF | 关 | **90.28%** | **92.39%** | 77.22% |
| Hybrid RRF | 开 | 80.56% | 84.78% | 76.67% |

默认选择 Hybrid RRF、关闭 metadata filter。当前 metadata 过滤损害召回，保留为实验选项。

## 24 篇故事端到端

| 指标 | 结果 |
|---|---:|
| 执行完整性 | 24/24 ok |
| Recall Extraction | **100.00%** (72/72) |
| Extraction hallucination rate | **0.00%** (0/81) |
| Recall@5 Retrieval | **89.58%** (43/48) |
| Accuracy Judge | **73.13%** (49/67) |
| End-to-End Precision | **66.67%** |
| End-to-End Recall | **75.00%** |
| End-to-End F1 | **70.59%** |
| 总故事处理时间 | 1703.11 s |

Judge 混淆矩阵显示主要问题为：8 条“不确定”被判成矛盾，5 条真实矛盾被判成不确定。结构性 partial/error 已清零。Benchmark 2.1C 的 50+ 篇扩容尚未开始。
