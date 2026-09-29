# 模块一：语义理解

本模块独立运行。输入小说短段落，输出全部语义事件和可追溯的JSON；不筛重要性，不查lore，不判断一致性，不写入设定。原Story-Level和Fact-Level benchmark仍冻结。

当前提示为understanding_v8。每次只提供目标片段和最多两个紧邻前文片段，不再在每个窗口重复完整story_text。编号只是引用单位，不代表每个片段都应生成独立事件。模型必须把每个目标编号归入事件依据或`non_event_span_ids`；后者只允许“随后”“做完这些”一类没有独立命题的连接/依附片段。位置/时间短语应随其修饰的谓语保留，不能为无谓语片段猜新机制。主体可以是物件/环境，不为物件额外猜出幕后控制者。inferred标签不允许无依据外推。v4-v7记录为历史观察，不能代表v8成绩。

## 内部结构

1. 程序给原文分句/分片段，保存字符起止位置。长片段按180字符继续切开，完整原文不截断；目前总输入≤800字符。
2. 每窗至多8个target片段、合计至多180字符，context只包含该窗之前最近两个原文片段。LLM将文学表达改写为直接事件，保留日常动作、环境、心理、否定、数量、左右、时间、条件和叙述性质。这个局部边界防止后续窗口随故事长度持续膨胀；远距离指代仍是已知限制。
3. 独立JSON包装按真实tokenizer测输入、在4096预算内预留至多1536输出；剩余不足256明确报错。JSON或顶层schema不合格最多重生成一次，保留每次raw和耗时；运行错误不反复重试。
4. 程序验证事件字段、原文编号与主体名称依据。mental_state/conditions/context_ids可省略，默认null/[]并记录defaulted_fields，不发明心理或条件。单个数组字段的文字可包成单项数组，已知字段大小写可以规范，记录wire_normalizations；不会猜测拼写错误字段、改变事实或替换语义分类。
5. 主体由模型提出。程序只为已有主体补最近的前文同名出现位置，记录unique/nearest_prior_name_anchor_added；这只是补名称引用，不证明代词理解正确，不更改事件。若找不到前文依据但窗内有候选，可调用一次仅补context_ids的LLM，再验证；无法补齐保留拒绝记录。
6. 未被事件或显式非事件声明覆盖的片段、被拒绝候选关联的有效原文片段，进行一次针对性补提；不替换已有事件。补提仍只带最近两个前文片段，并且只传递source_ids/context_ids与当前目标窗相交的拒绝候选，不把其他窗口的历史错误复制进来。仍缺少的内容完整列在coverage.source_spans及missing_source_ids。`non_event_source_ids`单列已审计的非事件片段；它们不会为了追求覆盖率而被强造为事件。补提若额外重复输出完全来自context、且没有声明替代候选的事件，只写入`extraneous_candidates`审计，不让无效窗外副本把已完成目标拖成partial。被拒绝候选保持原样，只有命题字段匹配或显式合法替代才能关联恢复后的E编号；未知分类被重提成合法分类时保留原记录，不算程序证明语义正确。
7. 按原文位置排序、分配E编号。只合并语义字段和原文位置完全相同的重复项；不同位置的同一句动作、不同条件或不同叙述性质都保留。同义表达不自动合并。

## 字段及状态

event.schema.json描述LLM wire输入输出中的events和non_event_span_ids；程序的完整report额外包含原文、定位、检查、调用记录。

- actors/event：主体和直接描述。
- mental_state：原文明示心理感受，无则null。
- explicit：是否原文直接表达；不是客观真假或确定程度。
- modality：observed（明示动作/状态/心理）、speech（声称）、belief（认为/猜测）、plan、dream、conditional（条件规则）、inferred（读者推断）。原文直接说“认为/计划/梦到”也是explicit=true，不能改成现实发生；只有额外推测才false/inferred。
- conditions：实际时间、先后及触发条件；source_ids/context_ids用于追溯。模型仍可能漏写这些内容。
- sources/contexts：程序还原的逐字原文及字符位置。
- review_required/review_reasons：指代仍不明、模型提出的代词解析、推断、默认字段等需复核事项。

status=ok只表示引用/结构/路由覆盖检查完成；partial表示缺口或未恢复的拒绝候选；error表示没有得到可用事件且所有调用失败。processing_complete不代表语义完整正确，semantic_verification固定为not_verified。原始被拒绝候选、重试和补提记录不会丢失。

失败窗口不能因后文把它当context引用就假装执行成功。unrecovered_failed_source_ids列出失败窗没有被合法事件source直接恢复的片段；每个失败调用记录recovered_by_primary_sources。任何未恢复失败都保持partial/error。request_seconds表示本次请求总耗时；model_loaded_this_request明确本次是否加载，model_load_seconds是共享模型实例首次加载的记录，不重复累加。

coverage区分primary_source_ids与context_only_source_ids。仅作上下文引用的片段仍应检查是否漏掉它本身的事件；所有片段被引用也不能证明没有漏义。模糊/文学/指代解析不可能用schema证明正确，错误留给人工复核及后续质量评测。

## 运行

补提允许每条修正事件携带replaces:["W1C2"]，显式说明替代哪个旧失败候选。程序要求编号确实属于之前拒绝、且新事件source_ids完整覆盖旧候选的有效原文依据；未知原文编号不能作为有效依据。原始拒绝和raw保留，修正事件标记recovered_candidate_reinterpreted供人工复核。没有显式替代或语义字段完全一致的恢复，不因某片段被其他事件引用而清除失败。failure_reasons列出unresolved_candidates、missing_source_ids及failed_source_ids。结构修正通过不表示语义已验证。

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline understand --text "雷喝了口水，看到德尔塔的名字时，他感到模糊的熟悉。" --device cuda --compact
```

`--file .\story.txt`读取UTF-8；`--output .\result.json`保存完整报告，不覆盖已有文件；`--compact`只缩减终端记录，文件仍完整。stdout是JSON，进度在stderr。无GPU用cpu；CPU实际性能未在本轮测量，不承诺速度。

API：

```python
from agent_pipeline import understand
from qwen_judge import QwenJudge

llm = QwenJudge('cuda')  # 本地固定版本模型只加载一次
first = understand('雷喝了口水。', llm=llm)
second = understand('他计划明天离开。', llm=llm)
```

两个请求互相独立，second不会借用first故事中的雷。多模块后续可以共享模型实例；本步只有understanding可用。可选progress回调接收window_id/purpose/target_ids。

真实观察测试：`python -m agent_pipeline.evaluation.run_understanding --device cuda`。12类测试独立保存报告，每条失败隔离，助手审阅与正式人工金标成绩分开。程序不让LLM自评、不用字面匹配宣称语义准确率。

本轮完整交付记录见[语义观察审阅](../../agent_pipeline/reports/module_understanding_20260918T071531Z_75a897/semantic_review.md)。真实CUDA运行9条ok、3条partial，这是工程处理状态；仍有时间条件、对白性质、心理幻觉和细节遗漏，语义质量未全部通过。review_required=false也不表示内容正确。下一模块才做事件筛选。
