# agent_pipeline：独立五阶段流程

当前模块完整说明及API见[语义理解模块](../docs/modules/UNDERSTANDING.md)，LLM协议见event.schema.json。采用understanding_v8提示、filtering_v7理由分类协议；理解窗只带最近两个前文片段，筛选窗只带前后各两个邻近事件摘要。系统保留原文引用、显式非事件片段、名称引用补齐、一次有限补提、可选字段默认与形状规范化、按来源排序/精确去重，并区分结构完成和语义待复核。筛选和段内检查对坏项做局部重试，不丢弃同批合法结果。历史实验与步骤说明保留供学习。

独立于冻结的Story-Level/Fact-Level benchmark。目前understanding、filtering、retrieval、judge、report可运行；[事件筛选说明](../docs/modules/FILTERING.md)、[检索说明](../docs/modules/RETRIEVAL.md)和[判断说明](../docs/modules/JUDGE.md)记录范围、状态和API。[报告整理说明](../docs/modules/REPORT.md)记录去重、证据和汇总边界，五阶段已接通，设计见[五阶段设计](../docs/planning/AGENT_PIPELINE_DESIGN.md)。

本模块已完成运行路径和错误处理交付；真实12类CUDA结果及语义问题见[本轮审阅](reports/module_understanding_20260918T071531Z_75a897/semantic_review.md)。9条ok、3条partial不代表语义准确率：时间条件、对白性质和心理幻觉仍需优化。

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline understand --text "雷喝了口水，看到德尔塔的名字时，他感到模糊的熟悉。" --device cuda --compact
```

无GPU可用`--device cpu`。也可`--file .\story.txt`读取UTF-8文本，`--output .\result.json`保存报告，已有文件不覆盖。模型来自现有本地缓存，不发送输入到远程服务；初次调用仍需加载模型。本步输入≤800字符。

流程：原文→程序编号/定位→小窗LLM语义理解→JSON及来源协议验证→一次有限补提→排序/精确去重→事件和覆盖/待复核报告。每窗约180字符、至多8片段，提供同段落前文上下文；多窗复用同一模型，格式或顶层schema失败最多重生成一次。原文不截断。字符偏移由程序保存，不让模型编造。

输出是JSON：events包含主体、直接事件表达、心理状态、明示/推断、叙述性质、条件、source_ids/context_ids，程序附上逐字原文及位置。calls保留每窗输入编号、原始回复、失败重试和推理指标。coverage指出没有被合法事件引用的片段；status=partial表示存在覆盖缺口、拒绝候选或失败窗，status=error表示所有窗失败。退出码0仅表示协议/覆盖检查通过，不代表语义准确性验收通过。

重要边界：喝水等动作在本模块保留，下一模块才判断是否需要与世界观核对。角色的熟悉感不能推出真的认识德尔塔；言外之意必须标为推断。片段被引用不保证其中所有事实都被提取，分类和推断标记也可能出错，仍须人工检查。本步不检索lore、不判断矛盾、不写入canon。

冻结旧benchmark的位置：benchmark_freezes/story_level_v1/manifest.json及snapshot.zip，包含旧源码/提示/资料/索引/已存Story报告。旧pilot仍为待审核草案，并无新增准确率成绩。

校验冻结内容：`& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline.verify_freeze`。

首个实际GPU例子保存到example_understanding_gpu_v1.json：模型保留喝水及看到名字时的熟悉感，引用覆盖三片段，JSON一次合法，生成7.76s（加载单列28.69s）。但是原文明示“感到模糊的熟悉”，模型错误标为explicit=false/inferred，且没有为第二事件选择含雷姓名的context。这是本步已发现的语义/指代问题，协议合法不证明理解正确；没有用程序偷偷改标签，也没有宣称完整性或准确性通过。

第二小步understanding_v2报告记录了熟悉感标记和引用补充失败，见[历史实验](../docs/history/UNDERSTANDING_STEP2.md)；这些是历史实验，不代表当前版本结果。当前版本真实12类测试通过`python -m agent_pipeline.evaluation.run_understanding --device cuda`运行，独立保存在本目录reports，程序状态与助手语义审阅分开，不让LLM自评或伪造准确率。
