# Benchmark 3.1: Generation Audit

Benchmark 3.1 is a read-only diagnostic experiment over the official Benchmark 3 trace. It does not optimize generation or rerun the model.

## Part 1

Part 1 reconciles the 24 story rows with all 130 LLM calls, verifies the 24/24 Benchmark 2.2 quality guard, and classifies all 31 retry calls. It reports real input/output tokens and Prefill/Decode/Total time for each category.

The audit distinguishes coverage recovery from other failed-attempt retries and Judge row retries. Judge calls keep `BATCH_ROW_RETRY` as their primary category, with their underlying validation or truncation cause reported separately.

## Deferred Part 2

Field-level output cost, tokenizer-based estimates, Extractor/Judge lean projections, and the final Generation Audit report are intentionally deferred until Part 1 is reviewed. No Part 2 CLI is exposed in this checkpoint.
