# Agent Pipeline v2 Judge Batching Benchmark

| Batch | OOM | Peak allocated GiB | Peak reserved GiB | Total Judge s | Facts/s | Judge Accuracy |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | no | 2.781 | 2.930 | 602.200 | 0.080 | 79.17% |
| 2 | no | 3.036 | 3.143 | 350.290 | 0.137 | 79.17% |
| 4 | no | 3.063 | 3.320 | 185.266 | 0.259 | 77.08% |
| 8 | no | 3.634 | 4.004 | 129.217 | 0.371 | 79.17% |

Judge time includes tokenization, padding, generation, decoding, validation, and retries. Model loading and retrieval are excluded.
