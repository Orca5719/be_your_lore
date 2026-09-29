# Benchmark 2.1B — Retrieval A/B

| Method | Metadata | Complete Recall@5 | Evidence Recall@5 | MRR@5 | Candidate Pool Recall | Mean Candidates | Ranking s |
|---|:---:|---:|---:|---:|---:|---:|---:|
| dense | off | 69.44% | 69.57% | 0.6609 | 100.00% | 48.00 | 0.0065 |
| dense | on | 69.44% | 69.57% | 0.6609 | 93.06% | 36.62 | 0.0221 |
| bm25 | off | 84.72% | 86.96% | 0.7778 | 100.00% | 48.00 | 0.0188 |
| bm25 | on | 77.78% | 82.61% | 0.7898 | 93.06% | 36.62 | 0.0346 |
| hybrid | off | 90.28% | 92.39% | 0.7722 | 100.00% | 48.00 | 0.0558 |
| hybrid | on | 80.56% | 84.78% | 0.7667 | 93.06% | 36.62 | 0.0718 |

Complete Recall requires every Oracle lore item for a fact to appear within Top-K.
