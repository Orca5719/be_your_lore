# Benchmark 4.2A：Extractor 约束解码

## 4.2C：vLLM 批处理与 n-gram 推测解码（2026-10-18）

入口：`python -m agent_pipeline_v4_2.vllm_benchmark`。A/B 入口不变。
本机 CPU 测试不证明 vLLM/CUDA 兼容或加速；真实运行由用户执行。
vLLM 使用独立 WSL2/Linux 环境，不安装进原 Windows `.venv`。

### 一次性准备

1. 在 Windows PowerShell 导出当前固定 revision 的模型：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.vllm_benchmark export-model
```

这会加载一次模型并将现有 NF4、double quantization、BF16 配置保存到 `.cache/benchmark_4_2C_nf4`，占用额外磁盘空间；不会生成故事、改原权重或联网下载。已有完整导出会验证后复用；失败留下 `.export-*` 目录便于诊断，不会覆盖已有目录。所有文件记录 SHA256，vLLM 只读取这个预量化模型，不做静默的现场重新量化。

2. 若尚无 WSL2 Ubuntu，可在管理员 PowerShell 手动执行 `wsl --install -d Ubuntu-24.04`，按系统提示重启并完成 Ubuntu 用户创建。已安装则跳过。开发过程中没有安装或修改 WSL。本次工具环境查询 WSL 返回拒绝访问，因此不能据此确定主机安装状态。

3. 在 Ubuntu/WSL 终端执行（Ubuntu 24.04）：

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
cd /mnt/c/Users/Xhang/Desktop/project/worldcheck
bash agent_pipeline_v4_2/setup_wsl.sh
~/.venvs/worldcheck-42c/bin/python -m agent_pipeline_v4_2.vllm_benchmark preflight
```

安装会联网下载 vLLM 及其依赖；模型读取本地导出。需 `nvidia-smi` 能在 WSL 内识别显卡，不需要在 WSL 安装单独的 Linux 显卡驱动。环境锁定 vLLM 0.18.0、bitsandbytes 0.50.2，实际 torch/transformers/CUDA/GPU/系统版本写入 manifest。`preflight` 只检查环境、导出完整性和冻结资料；后端是否能实际加载由下一步筛选验证。

### 先筛选，再完整实验

在上述 WSL 项目目录执行：

```bash
~/.venvs/worldcheck-42c/bin/python -u -m agent_pipeline_v4_2.vllm_benchmark run --phase screen
```

默认矩阵：batch 1/2/4 × n-gram 0/3/5，共九组，0 为同后端普通解码基线。每组用独立子进程加载模型，预热一篇，再运行 SL-P03、SL-P04、SL-010、SL-017 四篇；会保留全部实际输出和逐批 trace。`START` 显示配置，`RESULT_DIR`/`SUMMARY` 显示输出位置。筛选数据不能发布正式质量分数。第一个后端异常会停止矩阵，并保留 `worker.log`；OOM 不会自动减小配置后冒充原配置。

将筛选的 `SUMMARY` 发回分析后，可在**新结果目录**运行选定配置的 24 篇完整实验，例如：

```bash
~/.venvs/worldcheck-42c/bin/python -u -m agent_pipeline_v4_2.vllm_benchmark run --phase full --batches 1 2 4 --ngrams 0 3 5
```

可缩小批大小或候选集合，但 `--ngrams` 必须包含 0。`--gpu-memory 0.80` 是默认显存比例；修改需新目录。加载和预热不计入 Wall；每组不是同时跑。该实现复用原 StaticBatcher，将不同故事的独立请求作为一组交给 vLLM；不是把故事拼接进 prompt，也不是跨组持续补入新请求。实际平均 batch 会报告。固定关闭前缀缓存、chunked prefill，使用 eager 执行，避免第一轮混入额外变量；这不是 vLLM 最大性能配置。

所有组显式固定 `async_scheduling=False`：vLLM 0.18 原本会为普通解码与 CPU n-gram 选择不同默认值，不能把这个调度变化混入对照。

本版只报告可直接测量的 Wall、批调用时间、输入输出 tokens、实际平均 batch 和故事状态。没有 worker 级可靠测量的 TTFT、prefill/decode 分解、接受率和显存峰值保持未知，不用提交步推断接受率。跨后端输出不承诺逐 token 相同；同后端质量需审核后再比较。

### 续跑与审核

同一命令加 `--output <原RESULT_DIR>` 可跳过已完成组并续跑中断组。模型、代码、环境、数据或矩阵变化会拒绝续跑；中断组 Wall 标为空，不可与完整组比较。已完成但 partial 的组保留失败事实，不原地覆盖审核结果；需新目录重跑。

完整实验每组生成 `review.json`，所有事件先留空以避免继承此前发现错误的旧标签。填写 label/gold_ids/note 并清空 pending 后，在原 Windows 环境进行审核评分和冻结下游（WSL 输出就在同一 Windows 目录）：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.vllm_benchmark score --result-dir "<Windows RESULT_DIR>"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.vllm_benchmark run-downstream --result-dir "<Windows RESULT_DIR>"
```

下游保留原 Hybrid Retrieval、Judge v2.1 和 Report，可用 `--arms batch_1_ngram_0 batch_1_ngram_3` 只比较选定组。`QUALITY` 指向 E2E 报告。缺审核、文件摘要不匹配、提取未全部成功或筛选运行都会拒绝正式评分。下游异常/partial 的状态计数保存在质量 JSON，表格的 Complete 标为 False，命令退出码为 2，不能用 F1 单独宣称完整质量通过。

离线重建性能表：`python -m agent_pipeline_v4_2.vllm_benchmark summary --result-dir <RESULT_DIR>`。

官方接口依据：[vLLM 0.18 n-gram](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/n_gram/)、[预量化加载源码](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/model_executor/model_loader/bitsandbytes_loader.py)。

---

本分支从 Benchmark 4.1 建立。A 只比较 Benchmark 3 的单请求 Extractor 与同一模型、同一 prompt 和同一输出协议下的 XGrammar JSON 约束。两个实验组共用一次 Qwen3-4B-Instruct-2507 NF4/BF16 模型加载，各预热一篇，正式各跑 24 篇。窗口、修复适配层、重试和覆盖逻辑不变；Judge 不受语法约束。

Windows 上 XGrammar 的 CUDA 掩码需要 Triton。本机没有 Triton，因此本实验用 XGrammar 编译和维护语法状态，在 CPU 上施加其掩码，再把 logits 返回 CUDA 模型；这部分传输耗时计入 `generate()`。结果只代表这台机器和该实现，不代表所有约束解码后端。

在桌面仓库执行（4.2 位于仓库内的独立 worktree）：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
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
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative validate
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative run --device cuda --prompt-lookup-num-tokens 5
```

`run`打印`RESULT_DIR`和`SUMMARY`。每篇原子保存提取结果和逐调用trace，中断后可用`--output <原RESULT_DIR>`续跑；续跑的Wall不再作正式横向比较。正式表展示调用、重试、tokens、TTFT、Prefill/Decode/Wall、显存、提交步和待审核变化事件。新增事件要审核`prompt_lookup/review.json`；审核后再运行：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v4_2.speculative run-downstream --result-dir "<RESULT_DIR>" --device cuda
```

下游报告输出`QUALITY`，包含Extraction Recall/Precision、Hallucination、Overselect和E2E F1。`smoke`命令只运行一条短故事、在同一模型实例中比较两种生成方式，不计入正式结果。推测解码有可能因候选匹配率、量化数值路径或调度开销而变慢或改变输出；正式结果会逐案列出差异，不预设提速。
