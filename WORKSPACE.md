# 唯一工作区与搬迁记录

唯一工作区：`C:\Users\Xhang\Desktop\project\worldcheck`。后续代码、命令、测试及新实验报告均在这里进行；不再另建工作副本。

## 当前入口

- Benchmark 5代码：`agent_pipeline_v5/`
- 你跑完的5B筛选：`agent_pipeline_v5/reports/benchmark_5B_20261020T191204/summary.md`
- 历史4.2A/B报告：`agent_pipeline_v4_2/reports/`
- 4.2C代码已完整保留，实验仍暂停。
- 新的5A审计（桌面路径）：`agent_pipeline_v5/reports/audit_5A_desktop_20261020/`

## 历史路径映射

|原工作目录|现在的对应目录|
|---|---|
|`C:\Users\Xhang\.codex\worktrees\benchmark-5\worldcheck`|`C:\Users\Xhang\Desktop\project\worldcheck`|
|`C:\Users\Xhang\Desktop\project\worldcheck\.worktrees\benchmark-4.2`|`C:\Users\Xhang\Desktop\project\worldcheck`|

原始报告、manifest、审核账本中的路径保持原样，以保留字节级证据；查找时将工作区前缀按上表替换。过去运行的代码身份不能当作新位置的新运行身份，因此不原地续跑历史实验；完整已保存结果继续可读取与离线汇总，无需重跑筛选。

## 验证与保留

- 默认Codex位置只有Benchmark 5；两个实验工作区的代码和报告均整合到桌面主目录。
- 757个文件/备份条目逐一核对SHA256，包括已有同内容报告；85个报告文件新复制，4.2未提交代码和差异另外保留备份。
- 搬迁清单和4.2未提交原件：`.cache/workspace-consolidation-20261020/`，不作为工作区使用。
- 全量回归702项通过；桌面路径的5A审计、5B summary和validate-b通过。未运行GPU模型。
- 模型缓存、虚拟环境、Node.js、Anaconda和WSL均未搬动或修改。
