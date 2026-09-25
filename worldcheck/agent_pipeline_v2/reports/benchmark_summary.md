# Agent Pipeline v2 Benchmark Summary

## Headline Quality

| Metric | Value |
|---|---:|
| Conflict F1 | 0.4889 |
| Extraction Recall | 0.7500 |
| Extraction Hallucination Rate | 0.0000 |
| Retrieval Recall@K | 0.5417 |
| Judge Accuracy | 0.6364 |

## Error Taxonomy

| Error Type | Count |
|---|---:|
| EXTRACTION_MISS | 18 |
| EXTRACTION_OVERSELECT | 9 |
| JUDGE_FALSE_NEGATIVE | 6 |
| JUDGE_FALSE_POSITIVE | 10 |
| RETRIEVAL_MISS | 10 |

## Batch-8 Performance

| Metric | Median | Min | Max | Stddev |
|---|---:|---:|---:|---:|
| total_judge_seconds | 132.0877 | 131.5699 | 133.7154 | 0.9141 |
| facts_per_second | 0.3634 | 0.3590 | 0.3648 | 0.0025 |
| peak_allocated_gib | 3.6341 | 3.6341 | 3.6341 | 0.0000 |
| peak_reserved_gib | 4.0039 | 4.0039 | 4.0039 | 0.0000 |

Batch size: 8; repeats: 3.
