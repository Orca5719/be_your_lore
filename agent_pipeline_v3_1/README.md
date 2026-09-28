# Benchmark 3.1 Generation Audit

Benchmark 3.1 reads the frozen Benchmark 3 result and explains its retry and output-generation cost. It does not load the Qwen model, call `model.generate()`, or change any pipeline behavior. Part 2 loads only the pinned Qwen tokenizer to estimate field-level token cost.

## Commands

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 validate `
  --result-dir "<BENCHMARK_3_RESULT>"

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 audit-retries `
  --result-dir "<BENCHMARK_3_RESULT>"

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 audit `
  --result-dir "<BENCHMARK_3_RESULT>"

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3_1 summary `
  --result-dir "<AUDIT_RESULT>"
```

The audit creates a separate output directory containing:

- `calls_audit.jsonl`: all 130 calls reconciled with their story attempts.
- `retry_taxonomy.json`: aggregate retry counts, tokens, timings, cases, and Judge underlying causes.
- `retry_taxonomy.csv`: one row per retry call with preserved trigger errors.

Primary retry categories are fixed. Judge retry calls remain `BATCH_ROW_RETRY`, while `judge_retry_causes` records whether the failed row was caused by truncation, an invalid citation, or another underlying validation failure.

The full `audit` command also creates `generation_audit.json/.md`, `calls_audit.csv`, `field_token_breakdown.csv`, and `counterfactual_projection.csv`. Actual trace tokens remain separate from tokenizer estimates. Lean projections are labeled `counterfactual_estimate`; projected time is a linear conversion using observed decode throughput, not a measured optimized run.
