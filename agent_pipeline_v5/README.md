# Benchmark 5

5A离线诊断已实现。[PLAN.md](PLAN.md)记录分块计划，[REVIEW_5A.md](REVIEW_5A.md)记录逐例审阅及优先级。

无需安装依赖、不加载模型：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 audit --source-root "C:\Users\Xhang\Desktop\project\worldcheck"
```

打印SUMMARY，默认在本工作区agent_pipeline_v5/reports创建新目录；可加--output指定新目录。已有输出或会覆盖源输入的目录会拒绝；同一工作区内使用agent_pipeline_v5/reports保存新报告。历史报告只读，处理前后核对哈希。

产物：audit_summary.md、audit.json、call_costs.json、error_attribution.json、provenance.json、manifest.json、decision.md。自动归因只报告可观察边界，REVIEW_5A.md为助手语义复核，不代表用户批准gold变更；未知指标为null。

固定读取4.2A/B归档并复算审核口径，缺档或不一致会停止。旧早期报告目录链接指向源根之外时标未读；本次哈希不能补证缺失的历史原始性。

5B与5C1四篇筛选已由用户运行，正式质量尚待审核。5C1分析见[REVIEW_5C1_SCREEN.md](REVIEW_5C1_SCREEN.md)。长GPU实验由用户运行；Windows、4B和依赖不变。4.2C暂停，模型上限评估保留为待办。

## 5B 短输出协议四组实验

代码和离线测试完成；正式GPU筛选/24篇实验尚未运行，不宣称提速或质量改善。无需安装依赖，不用WSL，不导出模型；显式复用主仓库现有.cache/huggingface及data/index。

四组都以冻结StaticBatcher batch=1逐篇串行执行：original_plain、compact_plain、original_lookup5、compact_lookup5。一次加载4B模型，每组各预热一篇。短协议只省略mental_state=null、conditions=[]、context_ids=[]和由modality唯一决定的explicit；非空信息必须保留，事件id原本已由程序生成。其他提取窗口、修复、补缺、三出口和1536生成上限不变。

新适配层在原ExtractionRepairLLM之前补齐协议，随后走原校验和修复；wire_calls.jsonl保存模型回复与补齐结果，extractor_runs保留原流水线修复后的尝试。拒绝未知字段及无效模式，不把失败回复当空事件。原pipeline文件未修改。

### 先运行四篇筛选

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 validate-b
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -u -m agent_pipeline_v5 run-b --phase screen
```

validate-b不加载模型，只检查冻结数据/代码、缓存、版本和磁盘。run-b才加载GPU。筛选固定SL-P03、SL-P04、SL-017、SL-022，四组共16篇正式请求流程，另有每组一次预热；每篇包含原本的多窗口/重试，不能当16次LLM调用。完成后发最后一条SUMMARY路径；若失败发RESULT_DIR及报错，warmup失败看该组failure.json。不要在筛选跑完前复制中间SUMMARY当最终结果。

性能表包含完整状态、调用数、真实输入/输出tokens、Wall和待审核量；summary.json另存prefill/decode、故事延迟p50/p95、TTFT、显存峰值、重试/补缺。未测指标为null。推测模式提交chunk不是候选接受率。故事延迟包含当篇推理、解析和资源检查，整组Wall还包含正式检查点保存；加载与预热单列，不与4.2历史并发调度Wall直接对比。

资源门禁：开始要求输出盘与TEMP盘各至少5GiB空闲，每次模型调用及下一篇前检查剩余至少2GiB；报告预算512MiB，达到预算在下一次检查停止。检查不预留空间，外部程序也可能消耗磁盘。OOM会停止后续模型调用，保存失败故事并终止矩阵，不自动改配置重试。

### 续跑与完整实验

中断后同一命令加--output "<原RESULT_DIR>"。成功组验证哈希后跳过；中断组保留成功检查点，Wall置空，不能作为速度对照。成功故事必须有对应inference/wire调用日志；日志丢失、配置/代码/提示词/依赖变化、文件摘要变化都会拒绝续跑。有已记录partial/error或失败预热的组要使用新目录，保留原错误记录。

筛选结果经复核再运行完整24篇；完整对照仍跑同一四组，先forward再新目录reverse实现AB/BA。若只有部分路径值得保留，先记录决定再设计精简实验，当前不提供任意混搭以免漏基线。

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -u -m agent_pipeline_v5 run-b --phase full --order forward
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -u -m agent_pipeline_v5 run-b --phase full --order reverse
```

上述命令分别创建新目录。逆序不会改变单组故事顺序、输入或生成参数。普通与lookup分别对照短格式，不相乘收益；20%仅为计划中的实用筛选目标。

### 审核后评分和冻结下游

每组成功结束会创建review.json，所有事件先pending，不继承旧错误标签。审核每条label/gold_ids/note，清空pending后才能正式评分。筛选结果不能发布正式质量；完整实验缺故事、错误状态、过期事件签名、串组、缺日志都拒绝。

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 score-b --result-dir "<完整RESULT_DIR>"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -u -m agent_pipeline_v5 run-downstream-b --result-dir "<完整RESULT_DIR>"
```

score-b只在下游完整且绑定相同run/review/extraction后给E2E分数；没有下游时只给提取分数，Complete=False。run-downstream-b沿用Hybrid RRF、top5、Judge v2.1 batch8及Report，读取主仓库BGE缓存且local_files_only=True；--arms可以指定要复核的组。GPU运行仍由用户执行。

离线重建性能表：python -m agent_pipeline_v5 summary-b --result-dir "<RESULT_DIR>"。

历史gold争议保持原口径，不将5B与标注修正混成一种收益。筛选只作淘汰，最终结论依完整人工审核和同轮对照。

## 5C1 已观测字段错误的零调用修复

当前只做两条白名单：ignored_spans中的source_Ids改为source_id；events中的mentalological_state改为mental_state。目标字段已有值则保留原样报错；未知字段、主体、条件、否定、证据、check_reason均不猜测、不删除。无有效JSON也不打捞。真实回复和动作保存在wire_calls.jsonl，原窗口校验、重试/补缺照旧。

四组为original_plain、repaired_plain、original_lookup5、repaired_lookup5，沿用原协议。修复本身不调用模型，整个提取调用数仍由原重试/补缺决定。5B短协议尚无审核后质量，暂不组合；5C2原补缺已增量，待新证据再改。接受早期回复可能保留过选或幻觉，需要重新审核，不能把格式通过当准确率提升。

直接在桌面唯一工作区运行，无需安装、导出或WSL：

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 validate-c
& ".\.venv\Scripts\python.exe" -X utf8 -u -m agent_pipeline_v5 run-c --phase screen
```

validate-c仅检查环境，不加载模型。run-c默认新建本目录agent_pipeline_v5/reports/benchmark_5C1_时间戳，四组每组四篇，另有预热。运行后发最后SUMMARY；失败发RESULT_DIR和报错。同5B资源门禁与续跑规则，不能把历史5B输出作为5C续跑目录。

筛选复核后再用run-c --phase full --order forward与新目录--order reverse。summary-c、score-c、run-downstream-c的参数同对应-b命令；正式评分仍要求24篇、完整审核和绑定下游。离线回放可用：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 replay-c --result-dir "agent_pipeline_v5\reports\benchmark_5B_20261020T191204"
```

回放只读旧报告、核对文件摘要及冻结校验器，不重新运行模型或整条修复链。其通过数不能当成真实节省的调用数、速度或质量。离线验证记录见[VALIDATION_5C.md](VALIDATION_5C.md)。
