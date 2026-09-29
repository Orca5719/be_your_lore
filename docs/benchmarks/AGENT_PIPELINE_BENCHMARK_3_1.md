# Benchmark 3.1: Generation Audit

Benchmark 3.1 is a read-only diagnostic experiment over the official Benchmark 3 trace. It does not optimize generation or rerun the model.

## Part 1

Part 1 reconciles the 24 story rows with all 130 LLM calls, verifies the 24/24 Benchmark 2.2 quality guard, and classifies all 31 retry calls. It reports real input/output tokens and Prefill/Decode/Total time for each category.

The audit distinguishes coverage recovery from other failed-attempt retries and Judge row retries. Judge calls keep `BATCH_ROW_RETRY` as their primary category, with their underlying validation or truncation cause reported separately.

## Part 2

Part 2 loads only the pinned Qwen tokenizer and re-encodes saved JSON values. It reports measured output tokens separately from estimated event, coverage, verdict, citation, reason, assessment, and JSON-structure cost. Anything not explained by those estimates remains `transport_residual`; estimates never replace the measured trace.

The Extractor Lean Projection preserves every downstream event field plus ignored/non-event source identities. The Judge Lean Projection preserves every verdict and, only for contradictions, evidence identities and an 80-character reason. Malformed outputs are carried forward at their full measured token cost rather than treated as potential savings.

Projected token and decode-time reductions are marked `counterfactual_estimate`. They describe a theoretical upper bound under the observed decode throughput. They do not claim that a shorter schema would produce the same outputs, accuracy, latency, or retry behavior in a real generation run.

For a mixed Judge batch, saved per-row `generated_tokens` are used to isolate malformed or truncated rows. Valid sibling rows still contribute field and projection estimates. If those row-level measurements are absent or do not sum to the call total, the audit conservatively marks the whole call unprojectable instead of inventing an allocation.

Commands and output files are documented in `agent_pipeline_v3_1/README.md`.
