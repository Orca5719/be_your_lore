# Story-Level Consistency Benchmark：第一小步开发检查点

当前仅实现“短故事 → 重要原子事实”的可运行原型，没有检索、判断或保存；重要性筛选还未达到可靠验收，尚未建立24段正式评测集。

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 ".\story.py" extract --file ".\examples\story_step1\demo.txt" --device cuda
```
也可 --text "你的故事"；加 --json 获得所有事实、候选、筛选决定、引用和耗时。支持auto/cpu/cuda，本步骤最多800字符（更短学习示例允许）；严格检查已有4096 token预算，不截断。

方法：预编号带字符位置的原文片段；模型先生成事实候选；程序检查JSON字段、十种type、主体来源、时间来源和片段引用，程序还原source_text而非让模型抄原文。第二次模型调用只审查重要性，以keep/discard逐条覆盖全部合法候选，并记录丢弃原因。语义支持、重要性、指代和否定无法由这些结构检查保证。

公开事实有subject/predicate/object/type/source_text，并附id、source_start/source_end、source_ids、必要context与time。单个主体不明确或JSON字段错时独立拒绝，partial保留其余候选；整个JSON损坏、生成截断、重要性决定不完整则error，不伪装成功。空facts合法。原候选及原始输出保留供核对。提供候选wire JSON Schema声明，程序采用本协议专用校验器，不安装新依赖，不采用约束解码，不自动重试。

status=ok仅表示格式/引用及审查决定覆盖通过，不表示重要性筛选正确；unselected_spans只是未选片段，不能据此证明没有漏提。

真实GPU开发实验（不是benchmark分数）：
- 182字混合演示：最初提15条噪声，强化排除后全空，改用混合示范后提6条，继续修订提5条。新增重要性审查最初过度排除，修订后仍保留5条。其中3条确实关键：德尔塔获知双心脏、拉古艾尔治疗使右胸温暖、机械左臂断裂；放杯和回头两条仍属误提。原文时间和治疗条件已附context。各次输出保存为reports/story_step1_demo_*_attempt.json，最新完整结果为reports/story_step1_demo.json。
- 独立daily_only：空facts，符合预期。
- 独立explicit_vs_inferred：只保留芮尔戴导盲装置，没有补写失明或长期居住；另外2个候选缺object，状态partial。
- 独立important_eating：模型输出JSON后多了字符，解析失败；原始输出的object也为空。重要事实未可靠提取，状态error。
边界实验完整输出见reports/story_step1_controls.json。不能因单次演示调整成功/失败推断整体质量。这些故事是开发样例，不能宣称独立盲测。

验证：7项新增测试，全部78项回归测试通过；38个冻结版实现/提示/标注文件摘要均未变。已有Fact-Level代码、数据及报告没有改动。接下来应先处理输出协议与筛选边界，再接入检索判断，不能称第一步重要性任务已通过。


## 2026-09-17 状态事实协议修复 v2

- object允许缺省/null，旧空字符串规范为null；文字对象继续保留，非法对象类型仍拒绝。状态可以完整写在predicate，不人为编出对象。新增独立v2提示与schema；v1提示和旧实验不改。
- 主体必须能在引用的source_ids/context_ids中定位；缺少时，仅允许补回同一句里唯一含明确主体的先行片段，记录grounding_added_context_ids。跨句或多候选不自动猜，要求模型明确选context_ids。此检查只是文字依据，不证明语义指代正确。
- F编号对应原始候选位置，拒绝也使用相同F编号，不再保留后重新编号。
- 重要性审查缺项、重复或冲突逐条隔离。明确保留的进入facts；冲突/缺决定的放pending_facts和review_issues，状态partial，显示待复核；不替模型决定，也不将待复核条目当作确认事实。整体JSON格式损坏仍明确error。
- 银臂故事真实GPU输出：四个候选均通过新状态/引用结构校验，液化和渗透带主体依据；重要性审查把震中图像同时保留/丢弃。用同一原始输出回放新隔离逻辑后，液化和渗透显示正常、图像待复核，而非整批丢弃。原始实测与回放分开保存为reports/story_state_fix_v2.json和reports/story_state_fix_v2_replayed.json，不能冒充重新生成结果。
- 独立“艾琳吃下毒蘑菇后双眼失明了”实测能保留object=null的失明状态；并不保证也提取了完整中毒因果。

验证：82项回归测试通过；38个冻结Fact-Level文件摘要未变。仍仅提取，没有检索、矛盾判断或写入canon。命令不变。


## 2026-09-17 重要性筛选与主体检查顺序修复

用户两段原文仍partial：普通战斗动作因引用不足提前被拒绝；重要性模型批量输出重复冲突决定。新增三项回归测试，修复前两项失败，修复后全套85项通过。

- 候选结构/编号校验后，先决定重要性。被忽略的动作不因主体引用不足拖累状态；保留的事实仍严格检查主体依据，缺失则拒绝并partial。decode_facts默认仍严格；仅提取流程显式延后这项检查。
- 重要性审查每次仅一个候选，模型只返回decision和reason，由程序绑定编号，不能同时保留和丢弃。单次格式失败隔离为待复核，其他候选继续；保留每次原始输出和计时。不重试，不伪造决定。
- 首次实测仍误将“举起剑”借用邻近幻化能力判为重要，银臂普通屏幕变化也误留。保存首次报告后，审查改为仅当前事实及选定必要上下文，不再输入整段故事。加入当前动作不能借用旁边能力的重要性原则。
- 尝试补充因果分类边界时模型输出JSON后附说明，导致解析error；保存second_attempt报告，撤回该提示实验。最终仍使用story-extraction-v2，重要性提示为story-importance-v2。旧Fact-Level冻结版均不改。银臂最终仍多保留“屏幕开始发生变化”，渗透动作仍存在causality误标；这是尚未解决的重要性/语义分类问题，不将格式ok当成语义全部正确。

真实本地GPU复测（同一模型实例复用，下面耗时不含首次模型加载）：
- macklin: status=ok, 保留1条，忽略4条，模型已加载情况下端到端34.38s。
- silver_arm: status=ok, 保留4条，忽略0条，模型已加载情况下端到端27.31s。
- ordinary: status=ok, 保留0条，忽略0条，模型已加载情况下端到端1.90s。
- consequence: status=ok, 保留1条，忽略0条，模型已加载情况下端到端10.22s。

报告：reports/story_review_fix_v3.json；失败尝试：reports/story_review_fix_v3_first_attempt.json及second_attempt.json。最终两段原文重跑，其余两个边界案例沿用第二次实测，均标有prompt版本及每次调用耗时。只是开发检查，不是Story-Level正式benchmark成绩。逐条审查增加LLM调用数量，正确性优先，本步骤没有声称优化速度。未检索、未判断矛盾、未写canon、未建立24段正式评测。38个Fact-Level冻结文件SHA256未变。命令保持不变。
