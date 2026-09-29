| MetricBaseline | Value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Parameters | 4B (nominal) |
| Precision | NF4 4-bit weights, BF16 compute |
| Model VRAM (allocated after load) | Not recorded |
| Model tensor footprint | Not recorded |
| Peak VRAM (allocated) | 2.92 GiB |
| Peak VRAM (reserved) | Not recorded |
| TTFT (generation median) | Not recorded |
| Decode speed | Not recorded |
| Total latency (warm request median) | 8.76 s |
| First request (includes model load) | 22.77 s |
| Accuracy (all gold facts) | 82.29 % |
| Avg input tokens (per LLM call) | 635.95 tokens |

TTFT from model.generate start after tokenization; CPU token streamer synchronizes token delivery. Decode excludes first token; includes EOS. VRAM PyTorch process counters, GiB. Warm latency includes failed requests; accuracy includes missing facts.
