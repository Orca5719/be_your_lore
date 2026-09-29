# Agent Pipeline 2.1

该目录承载 Benchmark 2.1A/B/C，和冻结的 `agent_pipeline_v2` 分离。

当前只完成版本骨架。后续步骤依次接入 Judge 2.1、Oracle Retrieval、metadata-aware retrieval、BM25 和 RRF。稳定的模型加载、批处理和 v2 数据读取逻辑可通过显式导入复用，但不会修改 v2 默认行为或旧报告。
