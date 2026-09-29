# 模块四：逐事件一致性判断

输入第三模块的完整检索报告，输出每个实质事件与候选设定的判断。使用本地固定版本Qwen与judge_v6独立提示；不改写事件、提取新设定、保存canon或汇总整段故事。旧Fact-Level/Story-Level benchmark保持冻结。

## 判定方法

逐事件发送语义字段、选定原文上下文、相关支持事件和该事件的检索设定（含标题路径）。完整故事作为story_context辅助区分场景/指代，不能作为canon证据；第三模块的embedding查询仍只使用当前事件及局部上下文。设定编号在本次调用内为L1/L2等，不把cosine score发给模型当置信度，不提供外部知识。外部检索报告若没有第一层原文，story_context为空，不能凭空补场景。

- consistent（明确吻合）：证据直接支持相同主体、时间、位置、条件和叙述性质下的命题。仅相关、没有禁止或只支持一部分不能算明确吻合。
- contradiction（明确矛盾）：适用条件下有无法同时成立的明确冲突。左右宿主对调、数量直接冲突等可以构成矛盾；未记载的新能力或新实体不能自动判矛盾。
- uncertain（不确定）：缺少证据、只是相关、主体/适用条件不明、推断不足或部分命题未被支持。已有设定也互相冲突时，不擅自选一边。

对白、信念、梦境和计划保留其性质，不能把梦到宿主换位变成客观换位。引用只证明证据文字存在，程序无法证明模型推理正确。

本版的保守边界：speech/belief/plan/dream/inferred固定uncertain，origin=program_nonactual_scope，不送模型按客观内容判真伪。尚不能核对专门约束梦境、声称或信念的设定，也不能宣称这些类型自动无矛盾。人物明确知道/不知道、失忆等observed事件仍正常判断；conditional规则也可正常核对。此限制由第一轮真实梦境/信念被误判成现实位置矛盾引出，原始失败保留。

门控依据第一层的modality；如果第一层把声称或梦境错标为observed，仍可能误判。本层不保证补救所有上游语义错误。

## 内部结构

1. 提前校验检索stage/status、对应筛选分区、事件内容、唯一检索事件和证据ID/原文。support_only路由必须与上游一致。

   外部报告不能用ok掩盖嵌套筛选partial/error；pending/ignored编号必须完整匹配上游，不能默认为空而隐藏未处理项。
2. support_only不独立判断；pending/ignored仍保留在上游报告中，不擅自给判定。正常空证据直接uncertain/origin=program_no_evidence，不加载模型。没有实质事件也不加载。
3. 有证据才加载模型。可传入已加载llm，与前两层共享；模型加载失败一次后记录所有受影响事件，不重复加载。检索失败不送给LLM，保留error与uncertain占位，不算正常判断。
4. 模型返回assessment/verdict/citations/reason，协议见agent_pipeline/judge.schema.json。assessment包含same_subject、evidence_applicable（true/false/null）、relation（direct_support/direct_conflict/insufficient）及assumptions数组。明确结论若存在额外假设、对象/范围未确认，或证据关系不对应，程序降为正常uncertain，保留model_verdict/model_reason/raw及program_evidence_scope_guard来源。真实tokenizer控制4096输入输出预算，最多768输出，预算不足明确失败，不截断原文/证据。

   evidence_applicable检查证据规则是否针对当前对象、是否满足其生效条件；不检查待比较的左右/数量/属性值是否相等。同一人的左侧宿主规则适用于其右侧宿主断言，位置不同可构成冲突。旧字段same_scope容易被误解为属性值也必须相同，已在v4中替换。明确吻合需支持完整命题；明确矛盾只需一项明示内容与适用设定互斥，不能因其他动作未记载就回避冲突。
5. 明确结论必须引用设定；每条L编号须属于本次候选，quote须是对应text中的连续逐字原文，不能伪造、拼接或引用事件当设定。完整schema/引用失败最多重生成一次；运行异常不重复请求，失败不清除其他事件结果。所有raw、错误、重试和生成指标保留。
6. 合法引用补上canonical chunk_id，保持原证据中的文件/行号/标题等可追溯信息。完整根报告携带retrieval及其前两层记录。

## 状态边界

合法uncertain是成功完成的判断，可以status=ok；与格式/加载/运行失败得到的uncertain占位不同，后者status=error且有origin/error。所有实质事件失败或上游error，根error；有局部失败或上游partial，根partial；否则ok。processing_complete是处理状态，semantic_verification始终not_verified。

没有overall_verdict。本阶段不能证明整段无矛盾，也不能保证第一层没有漏义。第五模块report才做去重/合并/整理，但仍须保留遗漏和失败的边界。CPU选项可用，本轮真实Qwen判断观察使用CUDA，没有CPU推理速度成绩。

补提中的replaces显式记录对失败候选的修正，原始拒绝不删除。failure_reasons.upstream_stages展开未完成层的原因；compact另外显示上游拒绝数与未恢复数，避免judge本层rejected_count=0掩盖上游失败。assessment仍由模型判断，程序只检查协议与结论的一致性；模型把错误对象标成true或把间接证据标成direct_support，仍可能误判。

## 使用

在项目根目录执行完整四层串联：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline judge --text "雷的右侧心脏中，亚巴顿开始躁动。" --device cuda --compact
```

已有完整检索报告，可跳过前三层：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline judge --retrieval-file .\retrieval_result.json --device cuda --compact --output .\judge_result.json
```

--filter-file从筛选报告继续检索/判断；--events-file从理解报告继续筛选/检索/判断；--file读取UTF-8故事。原始故事路径中Qwen加载一次供理解、筛选和judge复用。--retrieval-device可以单独设置embedding设备。--output不覆盖文件，compact仅缩减终端，文件仍完整。partial/error退出码2，正常uncertain不因此返回失败。

```python
from agent_pipeline import judge_events
from qwen_judge import QwenJudge

llm = QwenJudge('cuda')
judged = judge_events(retrieved_report, llm=llm)
```

隔离观察运行器：`python -m agent_pipeline.evaluation.check_judge`。八条助手编写事件与已有真实链路分开，不是人工批准金标或正式benchmark；不让模型自评准确率。若月城/右胸本轮完整报告存在，还会复用其真实检索报告单独复测judge，标记reused_actual_upstream并保存输入文件摘要，跳过重新提取。每次新建报告目录，保留旧运行；源代码/活动提示/协议摘要用于检查运行期间版本是否稳定。

Step45的[历史观察审阅](../../agent_pipeline/reports/module_judge_20260918T140836Z_9c7366/semantic_review.md)保留首轮梦境/信念失败、范围门控及左心脏躁动结论偏强的语义问题。这是judge_v1历史记录，209项测试是当时程序检查，不能代表当前提示的模型质量或故事全量验收。

Step46的[最终真实复测与遗留问题](../../agent_pipeline/reports/module_judge_20260918T144000Z_e9e346/semantic_review.md)：活动judge_v4与understanding_v6，217项程序测试通过，11个观察组工程状态ok、源版本稳定。月城五事件不确定；右胸亚巴顿明确矛盾。旧链路左侧宿主躁动仍判吻合过强，未冒充正式准确率成绩。

Step48活动judge_v6：uncertainty_code可选、仅允许missing_rule/ambiguous_reference/unknown_conditions/partial_support/conflicting_evidence/insufficient或null。正常uncertain用程序固定表述，model_reason/raw保留；未提供代码兼容为insufficient。输出reason_origin=program_uncertainty_code，不增加生成次数，不改变verdict/citations；unknown code有限重试后明确失败。明确结论的理由仍由模型生成，已有范围/引用校验不等于语义正确证明。

Step 51 通用结构修复与后续边界见[本轮说明](STRUCTURAL_FIXES.md)。
