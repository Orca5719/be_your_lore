# Benchmark 5 — 结构优化设计与分块实施计划

> For agentic workers: use superpowers:executing-plans task-by-task. 每块完成测试、人工审核与结果记录后才进入下一块；不要一次实现所有候选。

**Goal:** 在现有 Windows / Qwen3-4B 环境中减少提取生成和无效复查成本，同时改善最终矛盾检测。
**Architecture:** 新增 agent_pipeline_v5 实验入口，复用冻结的切句、模型适配器、检索及审核工具；通过独立协议适配和策略接口逐项对照。历史版本只读，候选不自动成为生产默认。
**Tech Stack:** 原 Windows .venv、Python、Transformers 4.57.6、现有 Qwen3-4B NF4/double-quant/BF16；不安装新依赖。
**Spec:** 本文件的目标、实验契约和验收规则构成设计；下方任务为实施计划。
**状态:** 2026-10-20 5A完成：离线复算、逐例复核、32项相关测试通过；5B代码已实现待用户GPU实验；5C–5E未实施。4.2C暂停。

## 工作区与参考证据

- 新工作区由 Codex 管理，基于提交 3161a6d；不复制 4.2C 未提交改动，不复制模型、索引或大报告。
- 参考流程：对话“规划语义世界观检索器”：分块实现 → 测试 → 用户跑正式实验 → SUMMARY 审核变化事件 → 冻结下游 → QUALITY 复核 → LOG 记录/局部提交 → 决定下一块。
- 历史源目录：C:/Users/Xhang/Desktop/project/worldcheck/.worktrees/benchmark-4.2。
- 4.2B 报告：agent_pipeline_v4_2/reports/benchmark_4_2B_20261017T033125Z_295aa1。
- 4.2A 对照：agent_pipeline_v4_2/reports/benchmark_4_2A_20261016T170530Z_ab70a2。
- 4.2B 已审核：70/72 recall，precision 93.33%，E2E TP/FP/FN=18/9/6；Extractor wall 443.48s。与 4.2A 的速度差是跨轮观察，不作 B5 的同轮基线。
- 旧 summary.json 内可能仍有待审核快照；优先读取签名校验后的 review.json、修正账本及 quality_reviewed，不直接相信旧中间摘要。
- 已咨询 CodeGraph；当前索引未覆盖主要新流水线，按已定位文件读代码，不把旧索引当完整架构。

## 全局约束

- 模型/revision/量化、故事、索引、模型生成参数冻结；不下载/导出模型，不装 vLLM，不动 WSL。
- 模型缓存、索引、历史数据用显式路径配置，只读复用；新工作区路径不能触发重新下载或复制。启动前核对实际 qwen_judge.ROOT 与缓存解析路径。
- 不改现有 .venv，不清缓存、不搬磁盘、不自动安装依赖。
- 本轮代码和测试可以自动完成；GPU 长实验仍由用户运行，每块给确实已实现的命令。
- 每块只改变一个因素；普通生成与 Prompt Lookup 分开记账，后者 batch=1。不得将 batch>1 和推测路径混在同一质量归因中。
- 正式模型输入不能读取 gold、人工归因、review 或最终答案；它们只供离线评分。数据路径分离并测试。
- 保存原始生成、调用类型、修复前后内容、证据、版本与文件哈希；禁止覆盖旧报告或通过删失败故事提高分数。
- 恢复运行保留结果，但中断的整组 wall 标不可比较。代码变更拒绝原地续跑，除非单独实现并审计非语义迁移。
- 本轮只改检测流程；时间轴、角色图谱及完整事件入库另立任务。不得把检测用的相关事件子集当作完整故事事实库。

## 共同实验契约

输入：24 篇固定回归故事、冻结 canon/index、原始与修正后的审核账本。另建 24 篇挑战集，覆盖 6 类每类 4 篇：时间/倒叙、言论/信念、条件/例外、实体消歧、知识传播、多事实/否定；每类 2 篇开发、2 篇留出。先审核标注再冻结，留出部分不得用于提示词示例或错误驱动调参；首次揭盲后若调参就不再称未见集。

运行身份：JSON 规范化后的配置、代码/提示词/数据/索引摘要、模型 revision、依赖版本、设备、种子、生成预算、批次组成。报告产物包括 manifest.json、extractor_runs.jsonl、inference_calls.jsonl、review.json、summary.md、quality_reviewed.md、decision.md。

资源门禁（保守操作阈值，不是模型实际最低需求）：结果盘/系统 TEMP 盘均检查实际空闲空间，少于 5 GiB 拒绝开始；单次报告预算 512 MiB，进入下一故事前检查；剩余空间低于 2 GiB 时保存已有检查点并停止。报告明确不预留磁盘。模型禁止联网自动获取；OOM 停止，不自动换量化/批大小后冒充原配置。阈值改动写入新 manifest。

计时：加载、预热单列；正式 wall 包含所有失败尝试、修复、回退、检索与 Judge。分开记录提取 wall 与全流水线 wall，单篇 p50/p95 与批吞吐。未知 TTFT、显存、候选接受率为 null，不推算成真值。模型计时需沿用已验证同步方式，避免异步 GPU 计时偏差。

评分：提取 recall/precision、hallucination/overselect/duplicate、修复/恢复率、覆盖补回收益；E2E TP/FP/FN/F1、待确认比例与逐例差异。pending/partial/error 必须显示且禁止正式通过。人工审核有事件语义签名，改事件必须重审；不得仅按 E1 等编号复用。

决策：单次回归不证明统计显著性。小样本筛选只淘汰明显失败；入围配置在24篇同轮成对跑，采用 AB/BA 两轮，收益相反则补第三轮而不是取最好值。时间优势必须与质量门槛同时满足。留出集只在最终组合决策时使用。

## 5A — 离线诊断与冻结基线（第一块）

Files: agent_pipeline_v5/{__init__.py,__main__.py,diagnostics.py,identity.py}；tests/test_agent_pipeline_v5_diagnostics.py。
Interfaces: audit_archives(source_root: Path, output_dir: Path) -> dict；build_identity(config: dict, paths: list[Path]) -> dict。CLI 计划为 python -m agent_pipeline_v5 audit --source-root PATH，尚未实现，不能作为现成运行命令。

- [x] 写测试：缺档、重复/缺失故事、过期签名、tuple/list 往返、修正标签、缺指标为 null；audit 路径不导入 torch/transformers、不加载模型。
- [x] 运行测试确认先失败；实现只读档案检查，拒绝输入输出同目录。
- [x] 统计98次调用按初提/格式重试/覆盖恢复分类的耗时与tokens；区分补回有效事件、重复、幻觉。trace 不足以对应时写 unknown，不凭调用顺序猜归因。
- [x] 对 9 FP、6 FN 建逐例账本：case/gold或event、原文、提取、检索候选、Judge输出、首个可证实错误、后续错误、证据路径、审核状态。按评分器实际误报单位映射，不能假定15个独立故事；多因/证据不足保留待定。
- [x] 比较 B 的漏检集合与修正 A 基线，不只比总分；历史源码哈希缺失或重放不一致记录为限制。
- [x] pytest tests/test_agent_pipeline_v5_diagnostics.py -q；使用真实历史报告离线运行，核对70/72、18/9/6及98次调用。出 audit_summary.md、error_attribution.json、call_costs.json、decision.md。
- [x] 记录 LOG 并局部提交；确定 B/C/D 的实际优先级，不先宣称根因已证实。

## 5B — 短输出协议（第一层）

Files: agent_pipeline_v5/{wire.py,extractor.py,runner.py,resources.py,prompts/extractor_compact.txt}；tests/test_agent_pipeline_v5_wire.py、test_agent_pipeline_v5_runner.py。
Interfaces: expand_event(row: dict, event_id: str, spans: dict) -> dict；run_extraction_arm(config: dict, cases: list[dict], model, output_dir: Path) -> dict。
References: agent_pipeline_v2/extractor.py split_spans/字段校验；agent_pipeline_v2_1/extractor.py ExtractionRepairLLM；agent_pipeline_v4/static.py StaticBatcher；agent_pipeline_v4_2/speculative_model.py。

- [x] 测试并实现独立 wire schema。必填 actors,event,source_ids,modality,check_reason；沿用原先已由程序生成id的机制，不将其算作新收益。explicit严格按原协议由modality推导；mental_state/context_ids/conditions省略只等同null/[]，不能猜语义。校验失败不默默补 observed。
- [x] 测试言论/信念/条件/否定、跨句代词、未知source、源/上下文重叠、空actor、截断JSON；保留主体、时态和条件语义，不能将缺信息变成“无条件事实”。输出原文仍由原span表还原，模型不数偏移。
- [x] 首个版本保持旧 ignored/coverage 协议、输入窗口、修复策略、生成上限不变；不顺带删除覆盖逻辑或硬砍输出预算。仅缩短事件表示，并留原始响应。
- [x] 测试资源检查、只读共享缓存、断点身份、逐故事原子保存、失败退出；用fake模型做真实适配链集成测试。
- [ ] 用户筛选四组：原/短协议 × 普通/lookup5，固定batch1；固定4篇仅筛选，正式24篇人工审核后跑冻结下游。
- [ ] 第一候选门槛：相对同轮相同生成模式，wall目标降低>=20%，gold命中不减、幻觉不增、E2E TP不减/FP不增。20%是实用筛选目标非保证；不足可记负结果，不升级默认。
- [ ] 输出 SUMMARY 后审核，再输出 QUALITY/decision，局部提交。短格式若损害Prompt Lookup，不把其收益相乘。

## 5C — 局部修复与增量覆盖（第二层）

Files: agent_pipeline_v5/{repair.py,coverage.py,prompts/repair_local.txt,prompts/coverage_delta.txt}；tests/test_agent_pipeline_v5_repair.py、test_agent_pipeline_v5_coverage.py。
Interfaces: repair_event(event: dict, issues: list[dict], context: dict, model) -> dict；recover_missing(context: dict, accepted: list[dict], model) -> dict。

- [ ] 根据5A真正浪费时间的调用类型选择先做C1局部格式/语义修复或C2覆盖增量。分别设arm，不把两者同轮绑定。
- [ ] 确定性修复只处理可证明不改变语义的格式问题；未知证据/无依据actor送修复或保留失败。禁止简单删错事件后标ok。
- [ ] 局部提示带出错事件、验证错误和完整必要上下文；上下文不足回退旧路径。一次局部修复后仍失败，可回退一次原窗口路径；总模型调用预算不得超过所选冻结基线的原上限，所有回退计时。
- [ ] 覆盖补查保留原触发条件，只改为返回新增事件；接收端确定性去重不合并否定/条件/视角不同事件。不能将source已引用等价于该句全部事实覆盖。
- [ ] 测试一源多事实、空增量、重复增量、截断、修复改错主体、上下文遗漏、预算耗尽。程序规则不能证明语义完整，所有新遗漏要人工审。
- [ ] 分别对照冻结父配置，统计每次补查的有效新增、幻觉、耗时及最终质量。不得新增回归漏检/幻觉；节省不足或回退率过高则不采用。
- [ ] 逐块测试 → 用户运行 → 逐案审核 → 冻结下游 → decision/LOG/提交。

## 5D — 证据和判定结构（第三层，依5A归因安排）

Files: agent_pipeline_v5/{evidence.py,judge.py,prompts/judge_scoped.txt}；tests/test_agent_pipeline_v5_evidence.py、test_agent_pipeline_v5_judge.py。
Interfaces: build_evidence_packet(event: dict, candidates: list[dict], spans: dict) -> dict；judge_scoped(packet: dict, model) -> dict。
References: agent_pipeline_v2_1/retrieval.py、agent_pipeline_v2_1/judge.py、agent_pipeline_v4_1/audit.py。

- [ ] D1固定Judge、改变证据组织；D2固定证据、改变Judge提示。若实际错误是候选漏召回，再单设D0检索arm，不把候选扩展藏进D1。
- [ ] 证据包保留事件原文、canon来源、候选事实和已明确存在的时间/条件/视角。原字段没有的信息标unknown，程序不从句子凭空补结构。
- [ ] 时间、实体、视角关系不确定时保留候选，不做激进硬过滤；无检索结果是证据不足，不是consistent。扩展上下文受原输入预算限制，超限显式记录，不静默截关键例外。
- [ ] Judge区分世界事实/言论/信念/计划；要求不可共存的依据、来源ID和短理由，映射现有verdict协议。不得将unknown算作正确通过。
- [ ] 测试死亡→复活例外、倒叙、谎言、同名、知识传播、无证据、互相冲突canon。测试无gold/review信息进入模型输入。
- [ ] 冻结同一批已审核事件做组件对照，再跑完整流水线。以TP/FP/FN及待确认数量选择，不能靠全部弃答降低误报。
- [ ] 只接纳逐例解释清楚的改善；若证据标注本身有错，另建修正账本，不覆盖历史。

## 5E — 组合验收与批吞吐

Files: agent_pipeline_v5/{evaluate.py,report.py}；evaluation/benchmark5_challenge_v1.json；tests/test_agent_pipeline_v5_evaluate.py。
Interfaces: score_run(cases: list[dict], extraction: list[dict], downstream: list[dict], review: dict) -> dict。

- [ ] 旧audit校验硬编码24篇；挑战集需独立通用校验器，不改历史评分器。以旧24篇验证口径一致，再测12篇留出和缺/重复标签。
- [ ] 只组合已经单独通过的改动；与原结构同轮AB/BA，结果不稳补一轮。记录父配置和组合交互，负结果保留。
- [ ] 交互延迟以batch1比较，批吞吐另测普通生成batch1/4/8（先小样本资源检查；OOM停止）。Prompt Lookup仅batch1。不复用历史371秒作为当前同轮基线，也不再次引入自写连续调度器。
- [ ] 24篇回归不得增加漏检、无依据提取或误报；挑战留出报告逐类计数和差异。任何质量/速度权衡明确写出，未经用户接受不切默认。
- [ ] 发布最终Pareto表：延迟、吞吐、总时间、提取质量、E2E、待确认、RAM/VRAM（无测量为null）、磁盘占用。批处理输出变动必须重审，不能用batch1质量代替。

## 后续事项：4B能力上限评估（已标记，暂不执行）

先人工正确事件→原检索/Judge，再人工事件+完整但不泄露答案的证据→Judge；若仍有稳定关键错误且结构优化收益停滞，再考虑同条件更大模型对照。要在未参与调参的故事上成立，计入复核成本和弃答率。这里只记项目待办，不下载7B、不运行oracle、不切模型。现有oracle工具先复用评估，避免重复建轮子。

## Review focus

1. 短协议的省略字段是否悄悄改变modality/conditions：5B语义回归测试。
2. 多事实同源被覆盖检查吞掉：5C一源多事实与逐例gold差异测试。
3. 时间/视角unknown被当成一致或过滤证据：5D无证据及例外测试。
4. 标签/指标跨轮污染或待确认刷分：5A签名与5E评分隔离测试。
5. 工作区换路径触发下载、输出盘爆满或恢复后虚假计时：5B资源/路径/续跑测试。

## 本次交付边界

已完成：读取参考对话、核对代码与历史结果、建立B5独立工作区、冻结本计划及待办。
5A执行完成；5B代码已完成，下一步由用户运行筛选；其余阶段待结果复核后推进。尚未宣称任何提速/准确率收益，未运行GPU。

## 5A执行记录（2026-10-20）
- 先观察到缺模块失败，再实现；集成测试发现新父目录磁盘检查问题，增加数据绑定测试后补齐缺哈希校验。
- 独立审查发现外链数据集跳过历史哈希，增加失败测试并修复为拒绝。32项5A/4.2相关测试通过（19项5A）。
- 真实最终报告：reports/audit_5A_20261020_final；源文件处理前后哈希一致，未加载模型。
- Ruling: 历史早期报告外链只记录未校验，mandatory数据集外链拒绝；不扩大读取范围。
- Ruling: 原协议已不生成id，原补查已只补missing targets；修正后续假设，保留5B候选字段和5C单独对照。
- 15项错误逐例复核在REVIEW_5A.md；7项存在gold/证据契约争议，不修改历史分数。


## 5B执行记录（2026-10-20）
- Ruling: 首轮只省可选空字段和explicit，不减少输入窗口/生成预算/三出口，也不做5C策略改变。
- Ruling: 四组以同一模型加载、相同串行story顺序与batch1执行，新的成对基线替代跨轮历史Wall；共享缓存显式传入，仅初始化时临时切换原Qwen根目录后恢复。
- 接口：run-b/validate-b/summary-b/score-b/run-downstream-b；下游保留原Hybrid/Judge/Report。
- 74项相关测试、独立审查修复、本地validate通过；具体证据见VALIDATION_5B.md。GPU筛选与24篇质量/速度尚未运行，后三个实验任务保持未勾选。
