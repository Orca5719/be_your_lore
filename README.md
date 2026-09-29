# WorldCheck

本地世界观检索与一致性工具。当前工作是独立的 `agent_pipeline`，语义理解、事件筛选、检索、逐事件判断和报告整理已有可运行基线；语义质量问题仍在审阅记录中。五阶段已接通，汇总仅针对已核对事实。

## 常用入口

- [语义理解模块：使用方法、内部结构和API](docs/modules/UNDERSTANDING.md)
- [事件筛选模块：筛选范围、三类决定和使用方法](docs/modules/FILTERING.md)
- [语义检索模块：证据、设备选项和使用方法](docs/modules/RETRIEVAL.md)
- [逐事件判断模块：三分类、引用校验和使用方法](docs/modules/JUDGE.md)
- [报告整理模块：去重、汇总、证据与失败分区](docs/modules/REPORT.md)
- [全部文档目录](docs/README.md)
- [开发日志](LOG.md)
- [当前模块真实测试与语义问题](agent_pipeline/reports/module_understanding_20260918T071531Z_75a897/semantic_review.md)

## 体验

在项目根目录运行：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline report --text "雷捂住右胸，感到寄宿其中的亚巴顿开始躁动。" --device cuda --compact
```

无GPU可改为 `--device cpu`。完整边界和文件输入说明见模块文档。

旧检索实验的运行方法保留在[第一阶段说明](docs/history/README_PHASE1.md)。文档中的命令均以项目根目录为工作目录。测试资料不代表正式canon。
