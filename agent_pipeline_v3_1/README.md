# Benchmark 3.1 Generation Audit

Benchmark 3.1 reads the frozen Benchmark 3 result and explains its retry cost. It does not load the Qwen model, call `model.generate()`, or change any pipeline behavior.

This branch currently contains Part 1 only: call reconciliation and retry taxonomy. Output-field analysis and counterfactual lean projections remain intentionally unimplemented until Part 1 is reviewed.

## Commands

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 validate `
  --result-dir "<BENCHMARK_3_RESULT>"

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 audit-retries `
  --result-dir "<BENCHMARK_3_RESULT>"
```

The audit creates a separate output directory containing:

- `calls_audit.jsonl`: all 130 calls reconciled with their story attempts.
- `retry_taxonomy.json`: aggregate retry counts, tokens, timings, cases, and Judge underlying causes.
- `retry_taxonomy.csv`: one row per retry call with preserved trigger errors.

Primary retry categories are fixed. Judge retry calls remain `BATCH_ROW_RETRY`, while `judge_retry_causes` records whether the failed row was caused by truncation, an invalid citation, or another underlying validation failure.
