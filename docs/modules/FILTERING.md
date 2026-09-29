# 模块二：事件筛选

本模块接收第一模块的语义理解报告，对每个事件给出keep（保留）、ignore（忽略）或review（待复核）。判断的是是否值得与世界观核对，不是是否正确、是否矛盾，也不判定应不应该写入canon。模型仍使用本地固定版本Qwen，未增加远程推理。

当前提示为filtering_v7。保留标准是“可能受特殊设定约束”或“可能影响故事”，任一成立即可；普通动作遵守常识性物理规律不构成保留理由。未知lore、不与邻近事件形成已证实因果都不是忽略理由。模型收到目标事件的原文引用及有限邻近事件摘要，选择reason_code类别，由程序生成reason说明，避免自由理由新增机制/死亡/因果。若保留事件将拟忽略事件的原文引用为上下文，后者作为support_only=true的引用依据保留，记录model_decision/model_reason/model_reason_code及required_by_event_ids；不证明因果，也不表示它本身具有特殊机制。未解析主体不再阻塞明确的routine/process_detail忽略；若模型要keep，则转为review，避免把不明确的主体送入后续判断。额外推断仍进入review，不能自动转成可靠事实。

## 方法和内部结构

重要性依赖故事语境，不能只靠喝水、能力等关键词。采用独立LLM调用：每批最多8个目标事件，只附前后各最多2个邻近事件的简短语义摘要。目标事件自身仍保留source/context原文，因此局部承接和审计依据都在；不再为每一批复制完整故事及其余全部事件。这能控制输入规模，同时仍可区分普通动作与紧邻的重要后果。跨越多个事件窗口的远距离依赖仍属于后续长文本状态跟踪范围。

LLM返回事件ID、决定和理由代码，协议见agent_pipeline/filter.schema.json。代码限定为routine、mechanism、knowledge_relation、state_time_space、consequence_support、unclear，分别对应日常、特殊机制、知识关系、状态时空、后果依据、待复核。决定必须与代码一致。它不能返回新事实；程序按ID取回原事件，主体、心理、条件、叙述性质和原文引用全部原样保留。对白、信念、梦境和计划不能因筛选而变成现实。

步骤如下：

1. 校验上游stage/status、唯一事件编号和基本事件字段，复制输入以避免修改第一层结果。目前最多160个事件。没有事件时不加载模型。
2. 按8个事件分批，附上窗口前后各最多2个邻近事件的简洁语义字段；邻近摘要不携带sources、contexts或复核元数据，也不发送完整story_text。两个模块可共享一个模型实例。
3. 复用按真实tokenizer检查4096预算的JSON包装，输出至多1024 tokens；不足256明确失败，不静默截断。格式/顶层协议错误最多重生成一次，运行错误不反复重试。完整尝试、raw和耗时保留。
4. 验证每条决定的字段和当前目标ID；漏答、非法决定、重复或冲突只把对应目标加入一次局部修复调用，同批合法决定立即保留。若模型把同一个合法reason_code原样复制到decision，程序可无歧义地按固定表还原keep/ignore/review，并记录`decision_from_reason_code`；代码互相冲突、未知或带额外字段仍拒绝。修复仍失败的目标才进入review。原始坏候选写入rejected并标出是否恢复；已知但属于邻近上下文的多余决定写入extraneous_decisions，不拖累本窗；完全未知ID仍是协议失败。
5. 上游标注主体未明的事件若被模型keep，程序改为review；若模型以routine/process_detail明确ignore，则允许忽略，避免普通连续动作大面积制造partial。推断事件不能自动ignore。程序保留模型原决定及保护规则。来源编号数组在加载前验证，不让非法外部JSON触发TypeError。
6. 输出所有原事件events、逐项decisions、selected_events、ignored_events、pending_events及完整understanding报告。忽略只是路由建议，数据不删除；待复核单列，后续不得当作无影响或已通过。

## 状态边界

filtering_complete仅表示本层所有事件取得合法非review决定、没有拒绝或调用失败。processing_complete还要求上游status=ok。上游partial/error不会被成功筛选洗成ok。

所有窗口失败时status=error；存在待复核、拒绝或上游不完整时partial；否则ok。semantic_verification始终not_verified：格式检查、程序测试和筛选完成不等于语义准确。第一层遗漏或误解的内容，这一层不保证能找回，不承担重新提取或纠正事实的职责。

partial/error仍返回非零退出码2，不会为了终端看起来成功而改成0。compact和完整报告都暴露failure_reasons：上游状态、待复核编号、拒绝数和失败窗编号；stderr也明确提示未完整通过。partial与模型崩溃/格式异常须按调用记录区分。

上下文按固定窗口构造，不随事件总数线性增长；单个事件自身若异常庞大，仍可能触发token预算错误并明确失败，不会静默截断。CPU功能入口保留，本轮实际模型观察在CUDA上运行，未测CPU性能。

## 使用

在项目根目录运行。直接输入故事，先理解后筛选，两阶段只加载一次模型：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline filter --text "雷喝下杯里的水，随即毒发倒地。他的左心脏中，亚巴顿开始躁动。" --device cuda --compact
```

读取故事文件用--file。已有第一层完整报告，可跳过重新提取：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline filter --events-file .\understanding.json --device cuda --compact --output .\filter_result.json
```

--events-file必须是完整理解报告，不是只含events的片段或compact显示。--output保存完整JSON，不覆盖已有文件；stdout为JSON、进度在stderr；partial/error退出码2。无GPU用--device cpu。

API：

```python
from agent_pipeline import understand, filter_events
from qwen_judge import QwenJudge

llm = QwenJudge('cuda')
understanding = understand('雷喝水后毒发倒地。', llm=llm)
filtered = filter_events(understanding, llm=llm)
```

独立模块观察运行器：`python -m agent_pipeline.evaluation.run_filtering`，本轮固定CUDA。助手编写的案例是观察资料，不是正式人工金标；既有Fact-Level和Story-Level benchmark保持冻结。真实结果与审阅链接在开发日志中记录。

最终[真实观察审阅](../../agent_pipeline/reports/module_filtering_20260918T074457Z_c6c443/semantic_review.md)：7条中3条ok、4条partial，这是处理状态，不是准确率。普通日常可忽略，因果前提和验证依据仍会误筛，原因可能加入输入没有的内容。语义质量尚未全部通过；本层request_seconds仅为筛选请求，不含故事理解时间。

下一模块才执行embedding retrieval；目前没有检索、judge、report整理或设定保存。

用户报告失败的修复复测见[诊断与对照观察](../../agent_pipeline/reports/reported_failure_fix_20260918T081119Z_a62ba4/semantic_review.md)。原问题句三项保留、独立普通日常三项忽略，均ok；11项新增回归通过。支持引用按有限固定点传递保全；自由理由不再接受。此两段quick观察不是全量准确性验收。

Step48当前活动提示filtering_v6：新增process_detail→ignore，明确独立事实与操作/展示过程的区别。必要指代/引用仍由既有依赖闭包保全为support_only，不独立检索/判断。最后呈现的重要具体线索仍保留；异常屏幕机制并不因为对象是屏幕而被忽略。v5历史观察不能代表v6筛选质量。

Step55当前活动提示filtering_v7：保留v6的分类语义，只把窗口输入改为目标事件加有限邻近摘要。旧v6真实观察是历史结果，不能当作v7质量成绩。
