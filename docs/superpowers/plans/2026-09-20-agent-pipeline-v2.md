# Agent Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build an isolated one-stage event extractor and true batched Judge benchmark for batch sizes 1, 2, 4, and 8.

**Architecture:** `agent_pipeline_v2` owns its schemas and orchestration, while reusing only the root BGE Retriever, pinned Qwen model resources, and lore index. The Judge fixture benchmark materializes 48 balanced fact/lore inputs and runs every batch size in a clean worker process.

**Tech Stack:** Python, PyTorch, Transformers, BitsAndBytes NF4, NumPy, unittest.

**Spec:** `docs/superpowers/specs/2026-09-20-agent-pipeline-v2-design.md`

## Global Constraints

- Do not modify v1 behavior or existing frozen snapshots.
- Use true tensor batching, not multi-fact prompt packing.
- Keep all 48 facts in every Judge Accuracy denominator.
- Do not silently truncate story, evidence, or prompts.
- Record all work in `LOG.md`.

---

### Task 1: V2 extractor schema and coverage validator

**Files:**
- Create: `agent_pipeline_v2/__init__.py`
- Create: `agent_pipeline_v2/extractor.py`
- Create: `agent_pipeline_v2/extractor.schema.json`
- Create: `agent_pipeline_v2/prompts/extractor_v1.txt`
- Test: `tests/test_agent_pipeline_v2_extractor.py`

**Interfaces:**
- Produces: `split_spans(text) -> list[dict]`, `validate_extraction(report) -> dict[str, dict]`, `extract_events(text, device='auto', llm=None, progress=None) -> dict`.

- [x] Write failing tests for the three exclusive dispositions and missing/overlapping coverage.
- [x] Run the focused tests and confirm failures are caused by missing v2 code.
- [x] Implement the schema validator and deterministic span splitter.
- [x] Add the one-stage prompt and bounded LLM orchestration with one retry.
- [x] Run focused tests and v1 freeze verification.

### Task 2: V2 retrieval adapter

**Files:**
- Create: `agent_pipeline_v2/retrieval.py`
- Test: `tests/test_agent_pipeline_v2_retrieval.py`

**Interfaces:**
- Consumes: a complete v2 extractor report and root `Retriever.search(query, k)`.
- Produces: `retrieve_events(extraction_report, retriever, k=5, progress=None) -> dict`.

- [x] Write failing tests for selected-event routing, source preservation, Top-K, and per-event errors.
- [x] Confirm the focused tests fail.
- [x] Implement retrieval without importing v1 filtering.
- [x] Run focused tests and extractor tests.

### Task 3: Batched generation transport

**Files:**
- Create: `agent_pipeline_v2/batch_llm.py`
- Modify: `qwen_judge.py`
- Test: `tests/test_agent_pipeline_v2_batch_llm.py`

**Interfaces:**
- Produces: `QwenJudge._generate_batch(message_batches, max_new_tokens) -> list[str]` and `call_json_batch(llm, requests, max_output, validators) -> (values, report)`.

- [x] Write failing tests for left padding, row alignment, mixed valid/invalid output, failure-only retry, token accounting, and OOM records.
- [x] Confirm the focused tests fail for the expected missing interface.
- [x] Implement minimal batch tokenization/generation in the shared model adapter without changing `_generate` behavior.
- [x] Implement v2 batch JSON validation and retry.
- [x] Run focused tests and all existing Qwen adapter tests.

### Task 4: Batched Judge stage

**Files:**
- Create: `agent_pipeline_v2/judge.py`
- Create: `agent_pipeline_v2/judge.schema.json`
- Create: `agent_pipeline_v2/prompts/judge_v1.txt`
- Test: `tests/test_agent_pipeline_v2_judge.py`

**Interfaces:**
- Consumes: v2 retrieval report and `batch_size`.
- Produces: `judge_events(report, batch_size=1, device='auto', llm=None, progress=None) -> dict` with per-item results and batch metrics.

- [x] Write failing tests for batching 1/2/4/8, evidence-bound validation, all-item denominators, retry propagation, and programmatic no-evidence handling.
- [x] Confirm failures.
- [x] Implement batching and strict per-row validation.
- [x] Run focused tests.

### Task 5: Balanced fixed Judge fixture

**Files:**
- Create: `agent_pipeline_v2/fixtures.py`
- Create: `evaluation/agent_pipeline_v2_judge_48.json`
- Test: `tests/test_agent_pipeline_v2_fixtures.py`

**Interfaces:**
- Produces: `build_fixture(source_dataset, index_metadata) -> dict` and `validate_fixture(data, root, index_path) -> dict`.

- [x] Write failing tests for exact 16/16/16 balance, 48 unique claims, deterministic selection, evidence inclusion, and fixture hashes.
- [x] Confirm failures.
- [x] Implement deterministic selection and lore mapping.
- [x] Materialize and validate the frozen fixture.
- [x] Run focused tests.

### Task 6: V2 story pipeline and quality scorer

**Files:**
- Create: `agent_pipeline_v2/pipeline.py`
- Create: `agent_pipeline_v2/report.py`
- Test: `tests/test_agent_pipeline_v2_pipeline.py`

**Interfaces:**
- Produces: `process_story(text, llm, retriever, top_k=5, judge_batch_size=1, progress=None) -> dict` and v1-comparable provisional quality fields.

- [x] Write failing integration tests for extractor-to-retrieval-to-Judge data flow and partial status propagation.
- [x] Confirm failures.
- [x] Implement the v2 pipeline and deterministic report.
- [x] Run focused tests.

### Task 7: Benchmark workers and aggregation

**Files:**
- Create: `agent_pipeline_v2/benchmark.py`
- Create: `agent_pipeline_v2/benchmark_worker.py`
- Create: `agent_pipeline_v2_benchmark.py`
- Test: `tests/test_agent_pipeline_v2_benchmark.py`

**Interfaces:**
- Produces: `run-prepare`, internal `judge-worker`, `run-judge`, `score`, and `run-all` commands.

- [x] Write failing tests for clean worker commands, metric formulas, batch-size coverage, fixed denominators, OOM aggregation, and Markdown output.
- [x] Confirm failures.
- [x] Implement worker measurement and parent aggregation.
- [x] Run focused tests with a fake backend.

### Task 8: Documentation, logs, and full verification

**Files:**
- Create: `docs/benchmarks/AGENT_PIPELINE_V2_BENCHMARK.md`
- Create: `agent_pipeline_v2/README.md`
- Modify: `LOG.md`

- [x] Document metric definitions, commands, output files, and provisional limits.
- [x] Run all v2 tests.
- [x] Run the complete test suite.
- [x] Verify both existing benchmark freezes.
- [x] Run a non-generating dataset/config validation.
- [x] Record exact results and provide the CUDA benchmark command.

