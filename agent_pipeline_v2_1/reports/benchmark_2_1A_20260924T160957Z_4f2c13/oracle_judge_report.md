# Benchmark 2.1A — Oracle Judge

同一批72条事实使用 Oracle Retrieval，比较冻结的 Judge v1 与 Judge v2.1。

| Metric | Judge v1 | Judge v2.1 | Delta |
|---|---:|---:|---:|
| Accuracy | 0.6528 | 0.1667 | -0.4861 |
| Macro-F1 | 0.5223 | 0.1951 | -0.3272 |
| Contradiction FP | 16 | 0 | -16 |
| Contradiction FN | 5 | 30 | 25 |
| Unsupported inference | 6 | 0 | -6 |

Status: partial
