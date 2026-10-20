# 4.2C validation — 2026-10-18

Implemented and checked locally; real vLLM/CUDA compatibility, speed and quality remain unmeasured.

- Focused regression: 26 passed, covering 4.2A/B/C, including 13 new C tests.
- Full suite: **637 passed, 2 failed**. Do not call this a green full suite.
- First full attempt also encountered sandbox TEMP permission errors. Re-running with TEMP/TMP inside the writable worktree removed those errors.
- CLI `--help` works. Windows `preflight` rejects execution with an actionable WSL/Linux message.
- Independent review checked upstream vLLM 0.18 code. Explicitly fixed async scheduling across arms, both original EOS IDs, review gating and downstream identity hashes.
- Long GPU tests, checkpoint export, vLLM installation, WSL setup and VRAM fit were not executed.

Remaining existing tests:

1. `tests/test_agent_pipeline_v3_3_judge_a.py::test_offline_summary_writes_four_way_table_without_model` — missing `agent_pipeline_v3_2/reports/benchmark_3_2A_20261001T151227Z_cf125b/judge_lean_summary.json`.
2. `tests/test_agent_pipeline_v3_3_judge_a2.py::test_offline_seven_way_report_does_not_load_model` — missing `agent_pipeline_v3_3/reports/benchmark_3_3A_20261003T154610Z_7305ef/judge_a_summary.json`.

These tests and their production modules were not changed. Historical reports were not fabricated or rewritten to force a passing result.

Reproduce focused checks in this worktree:

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 -m pytest tests/test_agent_pipeline_v4_2_vllm.py tests/test_agent_pipeline_v4_2.py tests/test_agent_pipeline_v4_2_speculative.py -q -p no:cacheprovider
```

See README.md for Windows export and WSL screening/full-run commands.
