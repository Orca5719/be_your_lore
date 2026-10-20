# Benchmark 5C1 验证记录

日期：2026-10-20。唯一工作区：C:/Users/Xhang/Desktop/project/worldcheck。

交付两条已观测字段别名的确定性修复，原协议/修复 × 普通/lookup5四组，CLI run-c/validate-c/summary-c/score-c/run-downstream-c/replay-c。5C2与LLM语义局部补丁未实施，裁定及代价见PLAN.md；原补缺和重试保留。旧pipeline源文件未改。

本轮未加载模型、运行GPU、下载权重、安装依赖或操作WSL。未得出实际速度和质量结论。

## 自动验证

- 新模块首次测试缺模块失败；实现后集成测试通过。
- 专项：`.venv/Scripts/python.exe -X utf8 -m pytest tests/test_agent_pipeline_v5_repair.py tests/test_agent_pipeline_v5_wire.py tests/test_agent_pipeline_v5_runner.py tests/test_agent_pipeline_v5_diagnostics.py -q -p no:cacheprovider --basetemp .pytest-tmp-5c-review-green`：83 passed in 8.47s。
- 全套：本工作区TEMP/TMP设为.cache/test-temp-5c，运行`.venv/Scripts/python.exe -X utf8 -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-5c-final --tb=short`：722 passed in 33.70s。输出.cache/5c-tests.log。
- validate-c：4篇、4组、复用本地缓存通过；不加载模型。输出盘及TEMP空闲约23.5GiB，高于5GiB启动门禁。
- git diff --check通过。

## 独立审查和修复

一次只读独立审查发现两项P2，均已用失败测试复现后修复：

1. 补缺回复回放错误地用父窗口校验。现按前一尝试记录的missing target及冻结算法恢复真实target/context；无法重建则unknown。test_replay_uses_missing_targets_for_recovery_attempt由失败变通过。
2. 回放身份未绑定实际校验器。现保存当前agent_pipeline_v2/extractor.py摘要，并核对源档案冻结的相同文件摘要，不依赖旧目录仍存在。test_replay_rejects_unbound_validator由失败变通过。

审查未发现白名单覆盖已有字段、吞语义、增加模型调用或绕过原补缺路径；无延期审查项。

## 只读历史回放

源：agent_pipeline_v5/reports/benchmark_5B_20261020T191204。

最终回放：agent_pipeline_v5/reports/replay_5C1_20261020T214746/replay.md。earlier replay_5C1_20261020T214319是审查修复前记录，保留作过程证据，不作为最终验证入口。

|基线|原回复数|命中别名回复|修复后窗口通过|无法重建窗口|
|---|---:|---:|---:|---:|
|original_plain|19|2|2|0|
|original_lookup5|18|1|1|0|

源清单、完成状态、调用绑定和文件摘要通过，处理前后摘要一致。回放只在归档的旧wrapper输出上修复别名再校验，不重新执行生成或整个链；不能将3条通过视为实际少3次调用。接受早期输出可能多保留过选/幻觉。

## 待用户执行

run-c --phase screen，查看同轮成对调用/tokens/Wall，逐条审核全新review.json。筛选通过后再24篇AB/BA、冻结下游和正式质量；未通过前不升级默认、不组合5B短协议。操作命令见README.md。
