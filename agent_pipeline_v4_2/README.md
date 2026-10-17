# Benchmark 4.2A：Extractor 约束解码

本分支从 Benchmark 4.1 建立。A 只比较 Benchmark 3 的单请求 Extractor 与同一模型、同一 prompt 和同一输出协议下的 XGrammar JSON 约束。两个实验组共用一次 Qwen3-4B-Instruct-2507 NF4/BF16 模型加载，各预热一篇，正式各跑 24 篇。窗口、修复适配层、重试和覆盖逻辑不变；Judge 不受语法约束。

Windows 上 XGrammar 的 CUDA 掩码需要 Triton。本机没有 Triton，因此本实验用 XGrammar 编译和维护语法状态，在 CPU 上施加其掩码，再把 logits 返回 CUDA 模型；这部分传输耗时计入 `generate()`。结果只代表这台机器和该实现，不代表所有约束解码后端。

在桌面仓库执行（4.2 位于仓库内的独立 worktree）：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck\.worktrees\benchmark-4.2"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2 validate
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2 smoke-a --device cuda
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2 run-a --device cuda
```

正式命令先打印 `RESULT_DIR`，每篇后原子保存结果和调用记录，完成后打印 `SUMMARY`。将 `SUMMARY` 路径发给我，我会逐案核对变化事件。运行中断后可加 `--output <原RESULT_DIR>` 续跑；该组 Wall 时间会标为不可横向比较。离线重建表格：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2 summary --result-dir "<RESULT_DIR>"
```

每组保存 `extractor_runs.jsonl`、`inference_calls.jsonl`、`summary.json`、`pending_event_review.json` 和 `review.json`。只给与冻结 Benchmark 3 事件语义字段完全相同的条目复用旧审核标签；新事件的 `label` 留空，绝不按 `E1` 等编号沿用。审完变化事件并清空 `review.json` 的 `pending` 后，才运行下游与正式质量评分：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2 run-downstream --result-dir "<RESULT_DIR>" --device cuda
```

下游复用当前冻结的 Hybrid RRF、Judge v2.1 batch 8 和 Report；两组在同一执行中对照。输出 `quality_reviewed.md/json`，包含提取 Recall/Precision、Hallucination、Overselect 与明确矛盾的 E2E TP/FP/FN/F1。当前代码与 Benchmark 4.1 相同，历史 Benchmark 3 的归档下游摘要已有不匹配记录，因此不能把 4.2 的端到端绝对分数直接宣称为 Benchmark 3 的逐字复现。

`smoke-a` 不计入正式结果。它只证明本机真实模型的 batch=1 队列路径可以完成一条故事，不证明 24 篇质量或速度收益。

## 4.2B：Prompt Lookup 推测解码

4.2B只改变Extractor的生成方式：Transformers从当前输入中查找重复的token片段，提出最多5个候选token，由同一Qwen模型验证。模型、greedy策略、prompt、schema、窗口、重试和下游保持不变；不加入第二个模型或语法约束。当前锁定的Transformers 4.57.6在assisted generation内部要求batch=1，所以本轮只测单请求路径，不把它与4A/4B多请求batch吞吐混为同一指标。

为节省一次约15分钟的原版重跑，4.2B只重跑推测解码组，复用4.2A已审核的无约束batch=1结果作参考。报告标记为**跨轮对照**；若差距接近运行波动，需要再做配对重复测量。

在本工作区运行：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck\.worktrees\benchmark-4.2"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative validate
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative run --device cuda --prompt-lookup-num-tokens 5
```

`run`打印`RESULT_DIR`和`SUMMARY`。每篇原子保存提取结果和逐调用trace，中断后可用`--output <原RESULT_DIR>`续跑；续跑的Wall不再作正式横向比较。正式表展示调用、重试、tokens、TTFT、Prefill/Decode/Wall、显存、提交步和待审核变化事件。新增事件要审核`prompt_lookup/review.json`；审核后再运行：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative run-downstream --result-dir "<RESULT_DIR>" --device cuda
```

下游报告输出`QUALITY`，包含Extraction Recall/Precision、Hallucination、Overselect和E2E F1。`smoke`命令只运行一条短故事、在同一模型实例中比较两种生成方式，不计入正式结果。推测解码有可能因候选匹配率、量化数值路径或调度开销而变慢或改变输出；正式结果会逐案列出差异，不预设提速。
