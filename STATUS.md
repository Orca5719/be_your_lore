# Agent Pipeline v2 Benchmark

**State:** executed; draft annotations with an assistant-reviewed ledger.

## Contents

- 24 Story-track cases with 72 gold facts: 24 each of consistent, contradiction and uncertain.
- A 48-item balanced Oracle-evidence Judge fixture, evaluated at fixed batch size 8 over three clean repeats.
- The latest complete story result, review ledger, quality score, attribution output and benchmark summary.
- The source, tests, data/index snapshot and design/implementation documentation necessary to inspect the result boundary.

## Latest recorded result

Story track: extraction recall **0.7500**, retrieval Recall@5 **0.5417**, pipeline-conditioned Judge accuracy **0.6364**, and finding-level conflict F1 **0.4889**. The independent 48-item Oracle-Judge track records accuracy **0.7917** and macro-F1 **0.7839**.

These values are not a final generalization claim: the dataset retains `annotation_draft_not_final_benchmark`, the review ledger is assistant-reviewed, and error attribution remains pending review.

## Reproduction

From `worldcheck/`, validate the non-model protocol with `python agent_pipeline_v2_benchmark.py validate`. Running the GPU benchmark additionally requires the separately acquired Qwen model; weights and local environments are intentionally excluded from this repository.
