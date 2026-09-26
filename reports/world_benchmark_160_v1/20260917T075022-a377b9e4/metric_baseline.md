| MetricBaseline | Value |
|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Parameters | 4B (nominal) |
| Precision | NF4 4-bit weights, BF16 compute |
| Model VRAM (allocated after load) | 2.48 GiB |
| Model tensor footprint | 2.42 GiB |
| Peak VRAM (allocated) | 2.92 GiB |
| Peak VRAM (reserved) | 3.03 GiB |
| TTFT (generation median) | 361.05 ms |
| Decode speed | 15.00 tok/s |
| Total latency (warm request median) | 8.53 s |
| First request (includes model load) | 17.86 s |
| Accuracy (all gold facts) | 82.29 % |
| Avg input tokens (per LLM call) | 635.95 tokens |

TTFT from model.generate start after tokenization; CPU token streamer synchronizes token delivery. Decode excludes first token; includes EOS. VRAM PyTorch process counters, GiB. Warm latency includes failed requests; accuracy includes missing facts.
