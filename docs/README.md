# 文档目录

当前使用先看模块说明；历史文档用于回顾，不作为当前版本操作说明。文档中的命令仍在项目根目录运行。

## 当前模块

- [语义理解：内部结构、JSON、API、运行方法](modules/UNDERSTANDING.md)
- [事件筛选：筛选范围、三类决定、API、运行方法](modules/FILTERING.md)
- [语义检索：查询构造、证据字段、CPU/GPU、运行方法](modules/RETRIEVAL.md)
- [逐事件判断：三分类、证据引用、失败边界、运行方法](modules/JUDGE.md)
- [报告整理：去重、汇总与处理状态](modules/REPORT.md)
- [当前真实GPU观察与语义问题](../agent_pipeline/reports/module_understanding_20260918T071531Z_75a897/semantic_review.md)

## 规划

- [五阶段pipeline设计](planning/AGENT_PIPELINE_DESIGN.md)
- [产品需求备忘](planning/PRODUCT_NOTES.md)

## Benchmark与指标

- [冻结的Fact-Level Consistency Benchmark](benchmarks/FACT_LEVEL_CONSISTENCY_BENCHMARK.md)
- [冻结的旧Story-Level pilot运行说明](benchmarks/STORY_BENCHMARK_RUN.md)
- [模型性能指标说明](reference/MODEL_METRICS.md)

## 历史过程

- [第一阶段检索器与学习实验](history/README_PHASE1.md)
- [旧结构修复说明](history/STRUCTURE_FIX_V2.md)
- [旧Story提取第一步](history/STORY_STEP1.md)
- [旧Story检索第二步](history/STORY_STEP2.md)
- [旧Story判断第三步](history/STORY_STEP3.md)
- [旧Story预算及输出修复](history/STORY_EXTRACTION_FIX.md)
- [语义理解v2实验](history/UNDERSTANDING_STEP2.md)
- [语义理解模块实施时的计划](history/UNDERSTANDING_COMPLETION_PLAN.md)

## 记录保存规则

开发进度统一写在[LOG.md](../LOG.md)。原始测试报告继续保留在各自reports目录，不混进说明文档。lore中的Markdown是世界观资料，evaluation中的Markdown是评测协议/案例，均保持原位。

本次只迁移说明文档。冻结文档内容保持不变，旧路径到新路径记录在 `benchmark_freezes/story_level_v1/document_relocations.json`；原manifest和snapshot.zip保持不变，冻结校验使用新位置核对原SHA256。历史日志中的旧路径保留为当时记录，查找现位置请用本目录。
