# SDD ledger — plan: docs/superpowers/plans/2026-09-26-agent-pipeline-benchmark-3.md

Ruling correction: 桌面开发副本不是 Git 仓库；正式发布仓库位于 `C:\Users\Xhang\Documents\Codex\2026-09-23\github-plugin-github-openai-curated-remote\work\be_your_lore`，已有各 Benchmark 的远端跟踪分支。Task 1–3 尚未迁入正式仓库；Benchmark 3 后续应从 `benchmark/2.1c-bm25-hybrid-rrf` (`71e141a`) 建立专用分支。

Pre-flight: Task 1 的 SystemConfig/RunIdentity/run-row 校验供 Task 3 系统适配器消费；字段必须在 Task 1 固定，Task 3 不另造协议。
Pre-flight: Task 2 的 score_system/attribute_first_failures 未来供 Task 5/6 消费；本轮只实现纯函数，不绑定 CLI 或审核生成器。
Pre-flight: Task 3 的标准化结果供 Task 4 配对矩阵消费；本轮固定顶层 stage_metrics/config/result/status 接口，不提前实现矩阵。

Task 1: complete (无Git提交；tests: `python -m unittest tests.test_agent_pipeline_v3_schema -v` → 5/5 pass；RED为ModuleNotFoundError，GREEN为5/5)。

Task 2: Ruling: 报告遗漏必须由gold_findings/finding_mapping显式判断，不能把Judge正确判矛盾直接当端到端完成 — 符合spec的report_miss层 — 若错误会高估端到端Recall。
Task 2: complete (无Git提交；tests: `python -m unittest tests.test_agent_pipeline_v3_scoring tests.test_agent_pipeline_v3_attribution -v` → 7/7 pass；RED为缺模块及report_miss KeyError，GREEN为7/7)。

Task 3: Ruling: 当前项目 `.venv` 未安装pytest；旧pytest风格回归改用已安装pytest的D:\anaconda解释器运行，生产代码/模型环境仍使用项目`.venv` — 若环境差异影响测试，代价是依赖型测试可能漏报；本轮测试均不加载torch/transformers。
Task 3: complete (无Git提交；Benchmark 3自身17/17，关联v2/v2.1合计51/51 pytest通过；compileall通过；旧冻结260文件changed=[]、archive_ok=true)。

Final review: self-review（多代理未获本任务授权）；未发现Critical/Important/Minor项。Task 4–7按用户要求暂不实现。

