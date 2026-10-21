# Benchmark 5D0 验证记录

## 2026-10-21 格式遵循修订

用户报告reports/benchmark_5D0_20261020T235714：original_plain四篇完成，semantic_plain预热partial，后两组未运行。预热7次生成中4条回复（21个event对象）缺check_reason，涉及窗口1/3；共16个target未完成，耗时194.126秒。推理profile正常完成、输出未到1536上限，未见OOM；这是协议遵循失败，旧版完整四组速度/质量结论不可用。

原提示文件摘要与该次manifest一致。移除完整JSON示例可能削弱格式遵循，是待验证诱因，不是已证明唯一根因；过选/范围理解也尚未证明改善。

本次仅修改语义提示：开头恢复含全部9字段的JSON占位示例，check_reason置首，明确其无默认值；结尾检查字段。示例不带回归人物/设定事实，不修改程序validator或生成预算，不通过删事件/默认mechanism掩盖错误。

新增示例回归测试先失败后通过：用冻结_decode_window验证完整JSON示例。缺check_reason的脚本回复仍会按原预算失败、不生成事件。专项12 passed in 1.63s，validate-d通过且不加载模型。真实模型仍待用户重新运行，新提示哈希改变，必须新结果目录，不续跑或覆盖失败报告。

修订后全套：734 passed in 49.61s，命令`.venv/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-5d-format-final --tb=short`，TEMP/TMP使用本仓库.cache/test-temp-5d，输出.cache/5d-format-tests.log；git diff --check通过。

日期：2026-10-20。基准提交b2e3606，唯一工作区C:/Users/Xhang/Desktop/project/worldcheck。

## 实施边界

5A已指出知识状态/发言范围的协议混淆；5C1四篇中仍见日常过选和文本关系遗漏，因此先实现5D0提取语义契约。只替换system提示整体，四组original_plain/semantic_plain/original_lookup5/semantic_lookup5。不是逐条提示规则的消融，也不是Judge证据包改造。

原协议、切句/窗口、JSON校验、原repair wrapper、普通重试、missing-target补缺、1536上限、检索和Judge scope gate不改。不组合5B compact或5C1 alias修复。程序不猜测或修改modality，不用正则删日常事件。

新提示去除回归人物/设定示例，明确待核查性、source主体/关系/限定完整性、一源多事实与客观知识状态/通信发生和引语内容的区别。旧提示中的“双心脏”示例是可观察的干扰候选，不是已证明根因。新提示更长，不能预设提速；LLM语义理解只能由真实实验审核。

本轮不安装依赖、不下载/导出模型、不操作WSL、不加载真实模型、不运行GPU，不修改原始报告/审核标签。5D1/D2、5C2未实施。

## 测试证据

- 首次新测试9项全部失败，原因是prompt_policy/semantic arms/run-d入口尚不存在；实现后测试通过。
- 集成时一条fixture将逗号切出的S2事实误引为S1，覆盖校验正确拒绝；仅修正fixture，原验证器不变。
- 新10项测试覆盖：只替换system且不修改传入消息、raw输出不改/提示SHA256、禁止混合未验收策略、真实队列重试和增量补缺、一源多事实、belief/conditional保持、CLI dispatch与无模型包导入、新pending和screen正式评分拒绝、prepare不传gold且提示变更拒绝续跑。
- `.venv/Scripts/python.exe -X utf8 -m pytest tests/test_agent_pipeline_v5_semantic.py -q -p no:cacheprovider --basetemp .pytest-tmp-5d-identity --tb=short`：10 passed in 1.59s。
- 全套（TEMP/TMP设为本仓库.cache/test-temp-5d）：`.venv/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-5d-final --tb=short`：732 passed in 32.83s。完整输出.cache/5d-tests.log。
- `validate-d`：4篇/4组，冻结数据/旧代码、本地缓存、索引、版本、磁盘预检查通过；不加载模型。输出/TEMP均约24.3GiB空闲，高于5GiB门禁。
- `run-d --help`入口正常；git diff --check通过。

fake回复用于验证程序协议、队列和日志，不模拟模型是否能理解新规则，也不产生Accuracy/F1。

## 独立审查

按executing-plans执行一次独立只读最终审查，核对B/C兼容、四组映射/策略互斥、重试和补缺的提示转发、真实wire/提示身份、无gold输入、全新审核和下游scope约束；未发现需修复问题或延期项。审查未运行GPU/实模，语义质量与性能保留给用户实验。

## 用户实验

运行run-d --phase screen，结果默认agent_pipeline_v5/reports/benchmark_5D0_时间戳，完成发最后SUMMARY。四篇用于筛选，完整24篇需另跑AB/BA、全新审核及冻结下游。事件更少或observed更多不能作为单独胜出依据；要核对原文关系、否定/条件、多事实、知识/发言范围。原24篇已见，不能称留出集。
