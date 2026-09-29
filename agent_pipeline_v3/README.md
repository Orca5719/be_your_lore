# Agent Pipeline Benchmark 3

Benchmark 3 profiles the LLM workload of the frozen Benchmark 2.2 Candidate. It does not change extraction, Hybrid RRF retrieval, Judge v2.1, prompts, or verdict rules.

The formal run is fixed to the existing 24-story dataset, one excluded warm-up story, one measured pass, Top-5 retrieval, and Judge batch size 8. Each real `model.generate()` call is recorded separately, including retries and recovery calls.

## Commands

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 validate

& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 run --device cuda
```

`run` prints `RESULT_DIR` before loading the model. A full run is expected to take roughly the same order of time as the Benchmark 2.2 Candidate run (about 25 minutes on the development machine), plus small profiling and report overhead.

If the process is interrupted, resume the exact result directory:

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 run `
  --device cuda `
  --output "<RESULT_DIR>"
```

To rebuild the reports without running the model:

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 summary `
  --result-dir "<RESULT_DIR>"
```

## Outputs

- `inference_calls.jsonl`: one row for every real generation call.
- `story_runs.jsonl`: canonical per-story outputs and stage timings.
- `profile_summary.json`: machine-readable aggregate profile and quality guard.
- `profile_summary.md`: component, workflow, retry, slow-call, and alignment tables.
- `component_metrics.csv`: component aggregates.
- `inference_calls.csv`: flattened call trace for analysis.

The quality guard requires every semantic output to match the saved Benchmark 2.2 Candidate reference. A mismatch makes the run exit with code 2 and lists the changed story IDs in `profile_summary.json`.

TTFT starts after tokenization, immediately before `model.generate()`, and ends after the first completed generation step. Judge TTFT is one batch-level measurement. Token throughput is calculated from total tokens divided by total time; VRAM values are per-call maxima and are never summed.
