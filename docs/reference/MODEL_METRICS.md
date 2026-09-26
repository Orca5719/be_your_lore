
## LLM MetricBaseline 测量（2026-09-17）

world_benchmark.py 完成后会在独立报告目录自动写 metric_baseline.md，并在 summary.json 写 metric_baseline。原始每次LLM调用的 timing 增加 ttft_ms、decode_tokens、decode_seconds、decode_tokens_per_second。case增加peak_reserved_mib；summary增加模型加载后分配显存、权重与buffer张量占用及峰值reserved。

定义：
- Model VRAM：模型加载后当前进程torch.cuda.memory_allocated，不含生成KV cache；另列get_memory_footprint权重/buffer张量字节，不作为整张卡显存。
- Peak VRAM：完整160条请求最大PyTorch allocated，另列reserved。单位GiB，不是nvidia-smi整卡值。
- TTFT：每次generate在分词完成后开始计时至首个生成token，median；不包含模型加载、BGE检索、分词，不是网页首响应延迟。
- Decode speed：首token之后的token数总和 / 对应解码时间总和，包含EOS。不使用包含prefill的generation_throughput冒充解码速度。
- Total latency：去除首条请求的其余159条完整应用请求中位时间，包含提取、检索、多个事实判断及失败请求；首条含模型加载单列。
- Accuracy：verdict_correct / claims_total，保留遗漏和失败在分母；同时保留已有整体和全项指标。
- Avg input tokens：每次LLM调用平均prompt token数；不是每条用户输入长度，一条输入可能触发多次调用。
- TTFT使用轻量HF token streamer记录，prompt回调不算首token。HF为传递生成token会CPU同步，测量有额外开销；新旧速度非严格无扰动对比。未加并发、未实现用户流式界面。

已生成旧报告的可用指标表metric_baseline_from_existing.md；没测过的字段显示Not recorded，不能从总耗时反推TTFT。旧raw和summary不改。

命令保持不变：
```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 ".\world_benchmark.py" --device cuda
```
