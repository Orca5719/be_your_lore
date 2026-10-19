# Benchmark 5

5A离线诊断已实现。[PLAN.md](PLAN.md)记录分块计划，[REVIEW_5A.md](REVIEW_5A.md)记录逐例审阅及优先级。

无需安装依赖、不加载模型：

```powershell
Set-Location "C:\Users\Xhang\.codex\worktrees\benchmark-5\worldcheck"
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v5 audit --source-root "C:\Users\Xhang\Desktop\project\worldcheck\.worktrees\benchmark-4.2"
```

打印SUMMARY，默认在本工作区agent_pipeline_v5/reports创建新目录；可加--output指定新目录。已有输出或与源目录重叠会拒绝。历史报告只读，处理前后核对哈希。

产物：audit_summary.md、audit.json、call_costs.json、error_attribution.json、provenance.json、manifest.json、decision.md。自动归因只报告可观察边界，REVIEW_5A.md为助手语义复核，不代表用户批准gold变更；未知指标为null。

固定读取4.2A/B归档并复算审核口径，缺档或不一致会停止。旧早期报告目录链接指向源根之外时标未读；本次哈希不能补证缺失的历史原始性。

5B–5E尚未实施。长GPU实验由用户运行；Windows、4B和依赖不变。4.2C暂停，模型上限评估保留为待办。
