# 模块五：报告整理

> Step 49（2026-09-19）更新：`report` 默认在终端打印面向作者的中文反馈；`--compact` 不会把它切回 JSON。需要机器输出时加 `--json`，配合 `--compact` 可简化 JSON。`--output` 始终保存完整 JSON，并在中文报告末尾显示绝对路径；不指定则明确提示未保存。


输入judge完整报告，输出可追溯的JSON反馈。当前协议为agent-pipeline-report-v2。纯Python完成，不调用LLM、不追加判断、不重写设定，不增加模型生成次数。旧Fact-Level/Story-Level benchmark保持冻结。

## 如何整理

先验证事件、分区与检索记录一致，引用L编号和canonical chunk_id对应，原文确实来自相应设定。缺失记录、重复编号、假引用或成功状态掩盖失败均明确报错。

成功判断进入findings：原事件、结论、理由、原文来源及引用。只有语义字段、原文位置与局部上下文、结论、引用证据、判断来源和assessment全部相同时才合并，保留全部event_ids和不同理由。不同位置的重复动作、不同条件/否定/时间/主体不能仅凭相似合并；不做模型语义改写。同一证据可以支持不同问题，这些问题保持分开，证据原文及文件、行号、标题放evidence共享表，通过chunk_id关联。相似度留在完整上游检索中，不当置信度展示。

原文source_ids/context_ids也必须相同，尤其在外部报告没有偏移时不能只凭原文文字相同认定同一次事件。候选证据集合（编号、正文、来源）也须相同；正常不确定即使没有引用，候选不同仍保留为不同条目。不把query score差异当成新的事实。

failed_items单列执行失败；pending_events保留筛选待复核；ignored_events保留忽略项；support_events保留被引用的辅助动作及依赖关系；unprocessed_events保留上游error导致未送judge的事件；review_items列出原事件的复核标记。失败的uncertain占位不计入正常不确定数量。完整judge报告及所有前三层raw、输入与指标放根judge中，整理过程不删证据。

## 汇总边界

summary.scope始终checked_events，只概括已核对事实：

- 成功判断中有明确矛盾：显示“已核对事实中发现明确矛盾”，即使另有未完成项也保留这个发现。
- 没有明确矛盾，但有正常不确定、待复核、partial/error或没有实质核对事件：汇总不确定。
- 处理ok且实质事件全部明确吻合：显示“已核对事实明确吻合”。不证明整个故事无遗漏或无矛盾。

status与上游一致，processing_complete只是工程状态；正常不确定可以ok/退出0，partial/error退出2。review_required保留人工复核需要；optional_fields_defaulted等提示不等于执行失败。semantic_verification始终not_verified。summary.counts按原事件计数，finding_count是去重后条目数，二者不可混淆。

不确定可能来自设定未记载、检索未找全、指代或模型判断不明，不能自动宣称发现新概念，更不会写“新设定已记录”。上游误判会原样进入报告，例如已有观察中“亚巴顿在左心脏躁动”仍被仅凭宿主位置判吻合过强；本模块不会替judge修改结论。

## 使用

在项目根目录执行完整五阶段：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline report --text "雷捂住右胸，感到寄宿其中的亚巴顿开始躁动。" --device cuda --compact
```

已有judge完整文件，只整理，不加载Qwen或BGE：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline report --judge-file .\agent_pipeline\reports\right_host_judge_final_20260918.json --compact
```

--retrieval-file从第四层继续，--filter-file从第三层继续，--events-file从第二层继续；--file读取UTF-8故事。原始输入理解/筛选/judge共享一次Qwen，report不生成。--device cpu保留，无GPU可用；--retrieval-device单独控制embedding设备。--output保存完整JSON且不覆盖；--compact默认展示完整report、summary、evidence、复核标记与失败摘要，不再重复平铺findings和原事件。report中的excluded_details/processing_issues自包含原事实与主体；完整API和--output仍保留旧findings、各分区及全链路raw。

```python
from agent_pipeline import build_report
report = build_report(judge_result)
```

JSON字段约定见agent_pipeline/report.schema.json。此模块不更新旧benchmark，不新增正式accuracy/precision/recall/F1，不写canon。下一步可在五阶段接通后设计新的pipeline评测，先保留当前基线。

实际离线整理复测：`python -m agent_pipeline.evaluation.check_report`，不加载模型，新建报告目录保存输入内容摘要和源版本摘要；本轮最终目录agent_pipeline/reports/module_report_20260918T152430Z_89b397。完整五阶段CUDA报告已保存report_full_story_20260918.json，旧判断文件整理报告report_right_host_20260918.json与report_mooncity_20260918.json也保存。235项程序测试不等于模型语义准确率。

## Step48：完整报告结构及过程细节筛选修复

用户确认继续JSON格式。新增report对象前置完整反馈：overview整段核对总评；contradictions明确矛盾；uncertainties不确定；confirmations明确吻合；confirmation_topics需作者核实的事项（saved=false）；excluded_details被过滤/只作上下文的事实与原因；processing_issues失败、待复核、未处理及上游失败链。各项带原event/finding引用，不新增事实、不自作保存决定。非客观内容提示核实叙述性质，不直接建立现实设定。

filtering_v7沿用v6对独立剧情事实与过程呈现的区分：普通屏幕变化/文件切换用process_detail忽略，若为保留事实的必要引用则程序保全为support_only，不独立检索或judge；异常机制、重要状态、最终重要信息内容仍保留。不靠“屏幕”等关键词删除事件，断电仍显示已删秘密的异常仍可保留。v7只改变窗口上下文边界，不把结构改动宣称为语义质量提升。

judge_v6的正常不确定原因以受限uncertainty_code分类，由程序表达本次候选不足/对象不明/条件不明/仅部分支持/候选冲突，避免自由理由擅自引入左右、控制者或因果。model_reason/raw保留诊断；reason_origin明确为program_uncertainty_code。分类也是模型建议，不能证明全canon没有规则；未输出分类的旧wire回复兼容为insufficient。旧判断文件的旧理由不追溯改写。

最终实际观察与限制：[Step 48 审阅](../../agent_pipeline/reports/module_report_20260918T160809Z_a2821e/semantic_review.md)。

## Step 49：作者反馈

中文报告按“与已有内容矛盾”“可能新增的设定或剧情事实（待确认，未保存）”“需要确认”“明确吻合”展示。每项保留语义事实、原句、必要上下文、理由和已引用设定来源。处理失败仍单独提示，退出码保持0/2。

`possible_additions` 是正常不确定项的保守候选子集：仅 missing_rule 或 program_no_evidence，排除非客观范围门控。其他不确定不自动分类为新增；候选仍保留于原 uncertainties，原 verdict 不改变。kind=unclassified，须作者确认它是永久设定、一次性剧情事实还是检索遗漏，不推导永久能力、不保存canon。分类取决于已有模型诊断，不证明全库不存在设定。

这次没有增加模型调用、修改提示、人物/词语特判或检索、判断算法。保留记录重新整理的真实示例在 agent_pipeline/reports/author_report_mooncity_20260919.txt 和 author_report_right_20260919.txt；各有完整同名JSON。右胸宿主仍明确矛盾；月城震中图像列待确认可能新增，手臂液化/进入主机仍因已有诊断不充分列需要确认，没有为例句强行改结论。

原命令即可显示中文报告；保存示例：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m agent_pipeline report --text '你的故事' --device cuda --output '.\agent_pipeline\reports\my_report_02.json'
```

若需要原来的终端 JSON 输出，在末尾加 `--json`。

Step 51 通用结构修复与后续边界见[本轮说明](STRUCTURAL_FIXES.md)。
