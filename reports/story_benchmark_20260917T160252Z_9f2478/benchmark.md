# Story-Level Consistency Benchmark — pilot/draft

检测指标待语义配对审核，不能只按字符串或模型标签计分。

| MetricBaseline | Value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Parameters | 4B (nominal) |
| Precision | NF4 4-bit weights, BF16 compute |
| Model VRAM (allocated after load) | 2.48 GiB |
| Model tensor footprint | 2.42 GiB |
| Peak VRAM (allocated) | 4.21 GiB |
| Peak VRAM (reserved) | 6.77 GiB |
| TTFT (generation median) | 374.46 ms |
| Decode speed | 11.35 tok/s |
| Total latency (measured story median) | 10.83 s |
| First request (includes model load) | Not recorded |
| Accuracy (all gold facts) | Not recorded |
| Avg input tokens (per LLM call) | 956.06 tokens |

Explicit full-pipeline warm-up excluded from measured rows; all measured failures and retries included. TTFT starts after tokenization. Decode excludes first token, includes EOS. VRAM PyTorch process counters, GiB, includes Qwen+BGE; reserved may include warm-up cache. First-request latency is not measured; model/encoder load reported separately. Accuracy pending reviewed semantic matches.

| Story performance | Value |
|---|---|
| Model load | 15.865 s |
| BGE/index load | 0.610 s |
| Story latency P95 | 70.487 s |
| Sequential story throughput | 0.042 stories/s |
| Measured LLM calls (including retries) | 18 |
| Warm-up attempts / failures | 1 / 0 |

Stage status/counters: `{"case_status_counts": {"ok": 3, "partial": 1}, "extraction_status_counts": {"ok": 3, "partial": 1}, "retrieval_failed_facts": 0, "judgement_failed_facts": 0, "pending_candidates": 0, "rejected_candidates": 1, "retries": 0, "normalized_replies": 2}`
