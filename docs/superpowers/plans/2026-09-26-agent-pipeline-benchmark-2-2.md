# Agent Pipeline Benchmark 2.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Benchmark 2.2 that compares frozen Pipeline v2 against Hybrid RRF plus Judge v2.1 end to end and in a paired attribution matrix on the existing 24 stories.

**Architecture:** A new `agent_pipeline_v2_2` package adapts frozen v2 and tested v2.1 components without modifying them. It emits one normalized run-row schema, validates reviewed mappings, assigns one first-failure cause to each missed conflict, and renders auditable quality/performance comparisons.

**Tech Stack:** Python 3.12, PyTorch, Transformers, NumPy, existing BGE/Qwen runtimes, unittest/pytest-compatible tests.

**Spec:** `docs/superpowers/specs/2026-09-26-agent-pipeline-benchmark-3-design.md`

## Global Constraints

- Use the existing 24-story, 72-gold-fact dataset; do not expand it.
- Baseline is frozen Pipeline v2: one-step extraction, Dense Top-5, Judge v1, batch 8.
- Candidate is the same extraction family, Hybrid RRF Top-5 with metadata filter off, Judge v2.1, batch 8.
- Official device is CUDA; Top-K is 5; quality uses one deterministic run; performance uses one warm-up and three measured repeats with medians.
- Valid semantic errors are never retried; execution failures remain in denominators.
- Do not modify frozen v2, v2.1, earlier reports, fixtures, prompts, or freeze manifests.
- Every completed implementation step is recorded in `LOG.md`.

## Review Focus

- A gold conflict missing at extraction must receive only `extraction_miss`, even if later stages have no row; Task 2 pins this.
- A retrieved partial evidence set must remain `retrieval_miss`; Task 2 pins minimum-set semantics.
- A contradiction missed after complete evidence retrieval must be `judge_fn`, not retrieval failure; Task 2 pins this.
- Paired cells must consume byte-identical reviewed events; Task 4 pins event-digest equality.
- Resume must reject a changed dataset, prompt, index, method, Top-K, batch size, or model revision; Task 5 pins configuration hashes.

---

### Task 1: Benchmark 2.2 schemas and invariant validation

**Files:**
- Create: `agent_pipeline_v2_2/__init__.py`
- Create: `agent_pipeline_v2_2/schema.py`
- Create: `tests/test_agent_pipeline_v2_2_schema.py`

**Interfaces:**
- Produces: `SystemConfig`, `RunIdentity`, `validate_system_config(value)`, `validate_run_row(value)`, and `stable_digest(value)`.
- Consumes: JSON-compatible v2/v2.1 result dictionaries.

- [ ] **Step 1: Write failing schema tests** covering the fixed Baseline/Candidate configurations, forbidden Candidate metadata filtering, missing stage timings, duplicate case IDs, invalid statuses, and deterministic digests.

```python
def test_candidate_requires_hybrid_without_metadata():
    value = candidate_config(metadata_filter=True)
    with pytest.raises(ValueError, match="metadata filter"):
        validate_system_config(value)
```

- [ ] **Step 2: Run `pytest -q tests/test_agent_pipeline_v2_2_schema.py`** and confirm import/test failures.
- [ ] **Step 3: Implement frozen dataclasses/validators** with explicit enum values and canonical JSON hashing; no model imports.
- [ ] **Step 4: Re-run the Task 1 test file** and confirm all tests pass.
- [ ] **Step 5: Append Task 1 implementation and test evidence to `LOG.md`.**

### Task 2: Quality scorer and mutually exclusive failure attribution

**Files:**
- Create: `agent_pipeline_v2_2/scoring.py`
- Create: `agent_pipeline_v2_2/attribution.py`
- Create: `tests/test_agent_pipeline_v2_2_scoring.py`
- Create: `tests/test_agent_pipeline_v2_2_attribution.py`

**Interfaces:**
- Consumes: `score_system(cases, run_rows, review, k=5)` inputs compatible with reviewed v2 rows.
- Produces: extraction/retrieval/Judge/end-to-end metrics and `attribute_first_failures(...)` case/fact records.

- [ ] **Step 1: Write failing metric tests** for Precision/Recall/F1, zero denominators, Extraction Miss, complete minimum evidence sets, evidence micro Recall@5, MRR@5, Judge confusion, Judge FP/FN, and unsupported reasoning.

```python
assert score["end_to_end_conflict"] == {
    "tp": 1, "fp": 1, "fn": 1,
    "precision": .5, "recall": .5, "f1": .5,
}
```

- [ ] **Step 2: Write failing attribution tests** for the priority `execution_failure > extraction_miss > retrieval_miss > judge_fn > report_miss`, proving one primary cause per missed gold finding.
- [ ] **Step 3: Run both Task 2 test files** and confirm failures come from missing modules.
- [ ] **Step 4: Implement scoring by adapting, not editing, `agent_pipeline_v2.benchmark.score_story_quality`** and retain numerator/denominator counts beside every ratio.
- [ ] **Step 5: Implement first-failure records** containing case ID, gold fact/finding IDs, stage, evidence IDs, event IDs, and explanation.
- [ ] **Step 6: Re-run both Task 2 test files** and reconcile every aggregate with detail rows.
- [ ] **Step 7: Update `LOG.md`.**

### Task 3: Baseline and Candidate end-to-end adapters

**Files:**
- Create: `agent_pipeline_v2_2/systems.py`
- Create: `agent_pipeline_v2_2/story_pipeline.py`
- Create: `tests/test_agent_pipeline_v2_2_systems.py`
- Create: `tests/test_agent_pipeline_v2_2_story_pipeline.py`

**Interfaces:**
- Produces: `build_baseline_system(...)`, `build_candidate_system(...)`, and `process_story(system, text, progress=None) -> dict`.
- Baseline delegates to frozen v2 Dense/Judge-v1 behavior.
- Candidate composes `ExtractionRepairLLM`, `V21Retriever(method="hybrid", metadata_filter=False)`, `judge_story_events`, and `build_story_report`.

- [ ] **Step 1: Write failing adapter tests** using spies to assert Baseline selects Dense/v1 and Candidate selects Hybrid/no-metadata/v2.1, both with Top-K 5 and batch 8.
- [ ] **Step 2: Write failing normalized-output tests** requiring identical top-level fields, stage timings, token totals, status semantics, and retrieval provenance for both systems.
- [ ] **Step 3: Run Task 3 tests** and confirm missing adapters fail.
- [ ] **Step 4: Implement dependency-injected adapters** so unit tests do not load Qwen/BGE and production loads each shared model once per system run.
- [ ] **Step 5: Add CUDA metric hooks** that synchronize around measured stages and record peak allocated/reserved bytes without counting previous process allocations as incremental usage.
- [ ] **Step 6: Run Task 3 tests and existing v2.1 story-pipeline tests.**
- [ ] **Step 7: Update `LOG.md`.**

### Task 4: Paired two-by-two attribution runner

**Files:**
- Create: `agent_pipeline_v2_2/paired.py`
- Create: `tests/test_agent_pipeline_v2_2_paired.py`

**Interfaces:**
- Produces: `run_paired_matrix(extraction_rows, dense, hybrid, judge_v1, judge_v21, top_k=5, batch_size=8)`.
- Returns four named cells: `dense_v1`, `hybrid_v1`, `dense_v21`, `hybrid_v21`.

- [ ] **Step 1: Write failing tests** proving extraction is called zero times, all four cells receive the same event digest, retrieval changes only by cell, and Judge changes only by cell.
- [ ] **Step 2: Add failure-isolation tests** where one cell errors but the other three finish and the matrix becomes `partial`.
- [ ] **Step 3: Run Task 4 tests** and observe the missing runner failure.
- [ ] **Step 4: Implement retrieval caching per method** so Dense and Hybrid each run once per event set and feed both Judge versions.
- [ ] **Step 5: Implement per-cell quality summaries and deltas** for retrieval-only, Judge-only, and combined changes.
- [ ] **Step 6: Run Task 4 tests.**
- [ ] **Step 7: Update `LOG.md`.**

### Task 5: Resumable benchmark runner and CLI

**Files:**
- Create: `agent_pipeline_v2_2/runner.py`
- Create: `agent_pipeline_v2_2/cli.py`
- Create: `agent_pipeline_v2_2/__main__.py`
- Create: `agent_pipeline_v2_2_benchmark.py`
- Create: `tests/test_agent_pipeline_v2_2_runner.py`
- Create: `tests/test_agent_pipeline_v2_2_cli.py`

**Interfaces:**
- CLI commands: `validate`, `run-end-to-end`, `run-paired`, `score`, `summary`, `run-all`.
- Produces unique directories under `agent_pipeline_v2_2/reports/` and prints `RESULT_DIR=<absolute path>`.

- [ ] **Step 1: Write failing parser tests** for defaults CUDA/Top-5/batch-8/repeats-3/warmup-1 and all six subcommands.
- [ ] **Step 2: Write failing resume tests** for atomic JSONL rows, completed-case reuse, duplicate rejection, and hash mismatch rejection.
- [ ] **Step 3: Write failing lifecycle tests** for `pending_review`, `ok`, `partial`, and `error` without loading models.
- [ ] **Step 4: Run Task 5 tests** and confirm missing CLI/runner failures.
- [ ] **Step 5: Implement commands and manifests** including dataset, lore, index, model, prompt, source, and configuration hashes.
- [ ] **Step 6: Implement performance repeats** with warm-up excluded, median stage/total times, stories/s, facts/s, input/generated tokens, and CUDA peaks.
- [ ] **Step 7: Run Task 5 tests and `python -m agent_pipeline_v2_2 validate`.**
- [ ] **Step 8: Update `LOG.md`.**

### Task 6: Review ledger compatibility and complete scoring workflow

**Files:**
- Create: `agent_pipeline_v2_2/review.py`
- Create: `tests/test_agent_pipeline_v2_2_review.py`

**Interfaces:**
- Produces: `build_review_template`, `propose_review`, `reuse_exact_reviews`, and `validate_review`.
- Reuses v2 review semantics while adding system/track identity and unsupported-reasoning labels.

- [ ] **Step 1: Write failing tests** proving exact signatures may reuse mappings while changed actor, event text, source IDs, story ID, or system output cannot.
- [ ] **Step 2: Write exhaustive-ledger tests** requiring labels for every emitted event/finding and every unsupported-reasoning decision.
- [ ] **Step 3: Run Task 6 tests** and confirm failures.
- [ ] **Step 4: Implement review compatibility** and preserve `assistant-reviewed` provenance explicitly.
- [ ] **Step 5: Connect `score` to review validation, quality scoring, and first-failure attribution.**
- [ ] **Step 6: Run Task 6 tests plus existing v2 review/attribution tests.**
- [ ] **Step 7: Update `LOG.md`.**

### Task 7: Final reports, regression verification, and user-run handoff

**Files:**
- Create: `agent_pipeline_v2_2/report.py`
- Create: `agent_pipeline_v2_2/README.md`
- Create: `docs/benchmarks/AGENT_PIPELINE_BENCHMARK_2_2.md`
- Create: `tests/test_agent_pipeline_v2_2_report.py`
- Modify: `LOG.md`

**Interfaces:**
- Produces: `benchmark_2_2_summary.json`, `benchmark_2_2_summary.md`, per-case delta tables, and a freeze-ready manifest.

- [ ] **Step 1: Write failing report tests** requiring all requested metrics, absolute/relative deltas, configuration table, confusion matrices, first-failure totals, performance medians, limitations, and exact reconciliation links.
- [ ] **Step 2: Run Task 7 tests** and confirm the missing renderer failure.
- [ ] **Step 3: Implement deterministic JSON and Markdown renderers** that label model limitations separately from structural/execution failures.
- [ ] **Step 4: Run all Benchmark 2.2 tests, then the complete project test suite and compile check.**
- [ ] **Step 5: Run `python -m agent_pipeline.verify_freeze`** and require all prior freezes to report `changed=[]` and `archive_ok=true`.
- [ ] **Step 6: Run a no-model fixture smoke test** covering both tracks and report generation.
- [ ] **Step 7: Give the user one official CUDA `run-all` command.** Do not run the expensive official benchmark inside implementation unless the user asks.
- [ ] **Step 8: After the user run, review changed outputs, complete/reuse the ledger, run `score` and `summary`, then freeze Benchmark 2.2 only if all acceptance conditions pass.**
- [ ] **Step 9: Record commands, test counts, freeze verification, and remaining limitations in `LOG.md`.**

