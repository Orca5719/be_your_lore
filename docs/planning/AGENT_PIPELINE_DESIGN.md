# agent_pipeline：独立的五阶段流程

已获用户确认；2026-09-18起逐模块实现。复用本地固定版本Qwen及BGE，不改旧Story/Fact benchmark。

1. understanding：文字→全部语义事件，不做重要性过滤。主体、事件、心理状态、明示/推断、叙述性质、必要条件和来源；推断不冒充客观事实。输出events及来源覆盖缺口，覆盖不等于语义完整或正确。
2. filtering（可运行基线）：逐事件决定保留/忽略/待复核，针对受世界观约束或影响故事的事实；不改写、不删除理解阶段原始输出。协议和实际边界见../modules/FILTERING.md。
3. retrieval（可运行基线）：纯embedding查lore；保留片段来源和相似度。support_only保留为上下文不独立检索，pending仍待复核，见../modules/RETRIEVAL.md。
4. judge（可运行基线）：逐保留事实与lore核对明确吻合/明确矛盾/不确定，引用给定证据，失败与正常不确定区分。见../modules/JUDGE.md。
5. report（可运行基线）：整理、去重、合并，保留原始事件/事实/证据的追溯关系，不能自行改变判断。使用确定性程序，不追加LLM调用，说明见../modules/REPORT.md。

步骤一计划：冻结旧源代码/提示/资料/索引/已存Story报告并保存副本和SHA256；先写原文定位、覆盖缺口、推断标记、损坏JSON、超预算、分窗及部分失败测试；实现独立JSON调用包装、语义理解和CLI；运行回归与一个实际GPU体验例子；更新LOG。后续四阶段各自单独交付和讲解，本步不搭空壳假称完整pipeline。

语义理解wire协议：{"events":[{"actors":["雷"],"event":"喝了一口水","mental_state":null,"explicit":true,"modality":"observed","conditions":[],"source_ids":["S1"],"context_ids":[]}]}

modality：observed（叙述明确发生/状态）、speech（声称）、belief（认为/猜测）、plan（计划）、dream（梦境）、inferred（言外之意等推断）。explicit=false必须为inferred。modalities不能证明理解正确，仅避免混淆类型。程序补充ID、逐字sources、contexts和偏移；模型不生成原文文字或偏移。

输入目前≤800字符。按句/分句分窗，每窗约180字符、最多8片段，携带前两片段上下文供指代使用；不得只引用上下文产生新事件。一个长片段保持完整。输出预算受本地4096限制；不静默截断，失败窗保留错误和raw，其余继续。Qwen单次加载、多窗复用。JSON坏格式最多重生成一次，两个attempt均保留，不让重试替代合法但漏提的答案。
