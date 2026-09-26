# Agent Pipeline Benchmark 2.2

Benchmark 2.2 compares two complete story-consistency systems on the fixed 24-story dataset:

- Baseline: shared v2 extraction, Dense Top-5, Judge v1, batch 8.
- Candidate: shared extraction family, Hybrid RRF Top-5 with metadata filtering disabled, Judge v2.1, batch 8.

It also runs a paired matrix over byte-identical extracted events: Dense/Hybrid × Judge v1/v2.1. This separates retrieval changes from Judge changes.

## Run

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v2_2 validate

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v2_2 run-all `
  --device cuda `
  --top-k 5 `
  --judge-batch-size 8 `
  --warmup 1 `
  --repeats 3 `
  --output ".\agent_pipeline_v2_2\reports\benchmark_2_2_official"
```

`run-all` is resumable. It rejects reuse when the dataset, prompts, index, source adapters, fixed configuration, repeat count, or warm-up count changes.
The result directory is printed before model loading. Progress is printed per story, system, and LLM stage. The quality run is reused as measured repeat 1; each warm-up touches one fixed story per system and is excluded from medians.

After the run, review `review.json`. Set `provenance` honestly to `assistant-reviewed` or `user-reviewed`; every emitted event, finding, and reasoning-support entry must have a non-pending label. Then run:

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v2_2 score --result-dir ".\agent_pipeline_v2_2\reports\benchmark_2_2_official"
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v2_2 summary --result-dir ".\agent_pipeline_v2_2\reports\benchmark_2_2_official"
```

The final files are `benchmark_2_2_summary.json` and `benchmark_2_2_summary.md`. Model weights, Hugging Face caches, virtual environments, and local CodeGraph databases are excluded from Git.
