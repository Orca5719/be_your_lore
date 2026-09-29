# Benchmark 3: LLM Workload Profiling

## Purpose

Benchmark 3 explains where the frozen Benchmark 2.2 Candidate spends its time. It measures the existing 24-story workflow without changing model behavior or quality logic.

The profiled workflow is:

1. Extractor: one-step extraction of facts worth checking, including format retries and coverage recovery.
2. Retrieval: Hybrid RRF Top-5 with metadata filtering disabled; this stage has no LLM call.
3. Judge v2.1: batched consistency decisions with batch size 8, including failed-row retries.
4. Report: deterministic aggregation; this stage has no LLM call.

## Fixed protocol

| Item | Value |
|---|---|
| Quality baseline | Benchmark 2.2 Candidate |
| Model | Qwen/Qwen3-4B-Instruct-2507 |
| Revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Device and weights | CUDA, NF4, BF16 compute |
| Stories | Existing 24-story dataset |
| Retrieval | Hybrid RRF Top-5, metadata off |
| Judge batch | 8 |
| Warm-up | First story once, excluded and trace cleared |
| Formal sampling | One complete pass |

Dataset, quality reference, prompts, index manifests, model revision, and profiling source are hashed into the run manifest. Resume is rejected when any identity input changes.

## Call measurements

Every actual `model.generate()` produces one trace row. Retries are separate rows because they consume real compute.

- Useful input tokens are attention-mask tokens before padding.
- Padded tokens are the batch tensor size minus useful tokens.
- Compute input tokens equal useful plus padded tokens.
- `prefill_time` and TTFT run from generation start to the first completed generation step.
- `decode_time` runs from that point to synchronized generation completion.
- `total_time = prefill_time + decode_time` when a first token exists.
- CUDA peak allocated and reserved memory are reset and read for each call. CPU values are null.

The first-token stopping criterion returns `false` for every sequence and only timestamps the first invocation. It does not stop or alter generation.

## Aggregation

Component totals use sums for tokens and time, and maxima for VRAM. Throughput uses weighted totals:

```text
Prefill useful tok/s = total useful input tokens / total prefill seconds
Prefill compute tok/s = total compute input tokens / total prefill seconds
Decode tok/s = total output tokens / total decode seconds
```

TTFT p95 uses nearest rank. Retry cost is also reported separately. The workflow table contains model load, extraction, retrieval, Judge, report, LLM generation, non-LLM overhead, and end-to-end time. Timing alignment fields expose unexplained or double-counted time instead of hiding it.

## Quality comparability

The run stores complete semantic outputs, then compares a timing-free semantic digest against the official 24-case Benchmark 2.2 Candidate reference. Performance data is declared comparable only when all 24 cases match exactly. Any mismatch is reported by case ID and causes a nonzero exit status.

## Run

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 validate
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline_v3 run --device cuda
```

The result directory appears before model loading. Reusing it with `--output` resumes completed stories and preserves globally unique call IDs.
