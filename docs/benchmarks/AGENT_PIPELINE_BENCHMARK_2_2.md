# Agent Pipeline Benchmark 2.2

## Question

Does replacing Dense retrieval plus Judge v1 with Hybrid RRF plus Judge v2.1 improve story-level conflict detection, and which stage causes each remaining miss?

## Fixed protocol

- Dataset: 24 stories and 72 gold facts.
- Baseline: Dense Top-5, Judge v1, batch 8.
- Candidate: Hybrid RRF Top-5, metadata filter off, Judge v2.1, batch 8.
- Quality: one deterministic end-to-end run.
- Performance: one excluded warm-up followed by three measured repeats; report medians.
- Paired attribution: one shared extraction event stream, evaluated in four retrieval/Judge cells.

## Metrics

- Extraction Recall and Extraction Miss.
- Retrieval Recall@5 and Retrieval Miss using minimum evidence sets.
- Judge Accuracy, confusion matrix, FP, FN, and FN with complete evidence.
- Unsupported Reasoning rate.
- End-to-End conflict Precision, Recall, and F1.
- First failure: execution → extraction → retrieval → judge → report.
- Median total latency, stories/s, facts/s, input/generated tokens, and CUDA peak allocated/reserved memory.

## Audit rules

The review ledger must cover every emitted event, finding, and reasoning-support decision. Exact review reuse requires identical story/system/object signatures. `assistant-reviewed` is retained as provenance and never presented as author confirmation.

Execution failures remain in denominators. Semantic `uncertain` outputs are observations rather than execution failures. Earlier frozen benchmarks and prompts are read-only.

No official scores are recorded until the CUDA run, review validation, scoring, and summary steps all complete.
