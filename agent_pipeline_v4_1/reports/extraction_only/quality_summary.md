# Benchmark 4.1：CB2／CB4／CB8 质量复核

基于4B冻结提取输出，使用Benchmark 3原下游检索、Judge和Report。人工标签为assistant-reviewed；测试故事与gold仍是草案。

| CB | Gold命中 | Extraction Recall | Extraction Precision | Hallucination | Overselect | Duplicate | E2E TP/FP/FN | E2E F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 66/72 | 91.67% | 89.61% | 5/77 (6.49%) | 2/77 (2.60%) | 1 | 待运行 | 待运行 |
| 4 | 63/72 | 87.50% | 91.67% | 4/72 (5.56%) | 2/72 (2.78%) | 0 | 待运行 | 待运行 |
| 8 | 66/72 | 91.67% | 89.61% | 5/77 (6.49%) | 2/77 (2.60%) | 1 | 待运行 | 待运行 |

定义：Hallucination＝提取事实缺乏引用原文依据；Overselect＝原文支持但不值得设定核对；Duplicate＝已提取事实的重复条目。三类互斥，分母均为输出事件数。

E2E F1 以 gold 的明确矛盾为正类；无对应gold的矛盾判断也计FP，漏掉的gold矛盾计FN。

## 漏提 Gold

- CB2: SL-P03/G1, SL-006/G1, SL-008/G3, SL-012/G1, SL-015/G1, SL-020/G1
- CB4: SL-P03/G1, SL-006/G1, SL-008/G3, SL-010/G1, SL-010/G2, SL-010/G3, SL-012/G1, SL-015/G1, SL-020/G1
- CB8: SL-P03/G1, SL-006/G1, SL-008/G3, SL-012/G1, SL-015/G1, SL-020/G1
