# Agent Pipeline v2 Benchmark Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the existing Agent Pipeline v2 benchmark to 24 reviewed stories and add output precision, hallucination, retrieval-noise, finding-level detection, error-attribution, and fixed-batch-8 performance reporting.

**Architecture:** Keep the runtime pipeline unchanged and extend the benchmark around it. A strict dataset validator defines exhaustive gold annotations; a separate reviewed ledger labels every emitted event and finding; focused scoring modules compute stage metrics and earliest-stage attribution; the existing CLI orchestrates one story pass and three clean batch-8 Judge repeats into JSON and Markdown summaries.

**Tech Stack:** Python 3.12, unittest, JSON, pathlib, statistics, existing PyTorch/Transformers runtime.

**Spec:** `docs/superpowers/specs/2026-09-21-agent-pipeline-v2-benchmark-expansion-design.md`

## Global Constraints

- Keep this work in `agent_pipeline_v2`; do not modify v1 runtime behavior or historical result directories.
- Preserve the pinned Qwen revision, NF4 configuration, Judge prompt, BGE model, Top-5 retrieval, lore, and index snapshot.
- Formal Judge execution accepts batch size 8 only and runs three clean repeats.
- Story execution runs once; only reviewed mappings and output dispositions may be scored.
- Keep the current four stories as regression cases and add twenty stories.
- Target exactly 24 stories and 72 gold facts: 24 consistent, 24 contradiction, and 24 uncertain.
- Undefined ratios serialize as `null` and render as `N/A`.
- Do not add length bucketing in this milestone.
- The project is not a Git repository. Replace commit steps with a tested checkpoint and `LOG.md` update.

---

### Task 1: Expanded Dataset Contract and Validator

**Files:**
- Create: `agent_pipeline_v2/story_dataset.py`
- Create: `tests/test_agent_pipeline_v2_story_dataset.py`
- Modify: `agent_pipeline_v2/benchmark_cli.py`

**Interfaces:**
- Produces: `validate_story_dataset(data: dict, root: Path, index_dir: Path, *, expected_cases: int = 24, expected_facts_per_verdict: int = 24) -> dict`
- Produces: `load_story_dataset(path: Path, root: Path, index_dir: Path) -> tuple[dict, dict]`
- Consumes: current lore/index metadata and exact source anchors.

- [ ] **Step 1: Write failing tests for exact anchors and exhaustive fields**

```python
def test_gold_anchor_must_match_exact_story_slice():
    data = minimal_dataset()
    data["cases"][0]["gold_facts"][0]["source_anchors"][0]["start"] += 1
    with self.assertRaisesRegex(ValueError, "gold anchor"):
        validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)

def test_relevant_lore_and_minimum_sets_must_exist_in_index():
    data = minimal_dataset()
    data["cases"][0]["gold_facts"][0]["relevant_lore_ids"] = ["missing"]
    with self.assertRaisesRegex(ValueError, "lore"):
        validate_story_dataset(data, ROOT, INDEX, expected_cases=1, expected_facts_per_verdict=None)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 -m unittest tests.test_agent_pipeline_v2_story_dataset`

Expected: failure because `story_dataset.py` or the new validator is absent.

- [ ] **Step 3: Implement schema, anchor, ID, lore, and distribution validation**

The validator must require case fields `id`, `title`, `group`, `dimensions`, `story`, `character_count`, `gold_facts`, `ignored_events`, `non_events`, `gold_findings`, and `annotation_status`. Validate unique IDs; exact `[start:end]` anchors; non-overlapping intended dispositions; verdict/group totals; relevant lore IDs; minimum evidence subsets; finding references; corpus hashes; and index version.

- [ ] **Step 4: Make CLI `validate` use the new validator without loading Qwen**

Keep a temporary `expected_cases=4` compatibility mode until Task 2 publishes all 24 cases. Unit tests must pass in compatibility mode; formal defaults switch to 24 only in Task 2.

- [ ] **Step 5: Run focused and existing validation tests**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 -m unittest tests.test_agent_pipeline_v2_story_dataset tests.test_agent_pipeline_v2_benchmark`

Expected: all pass.

- [ ] **Step 6: Record the tested checkpoint in `LOG.md`**

Document schema version, validation boundaries, test count, and that runtime behavior is unchanged.

---

### Task 2: Author and Validate the 24-Story Dataset

**Files:**
- Create: `evaluation/story_benchmark_24_v2.json`
- Create: `tests/test_agent_pipeline_v2_story_dataset_real.py`
- Modify: `agent_pipeline_v2/benchmark_cli.py`
- Modify: `agent_pipeline_v2/README.md`

**Interfaces:**
- Consumes: `validate_story_dataset` from Task 1.
- Produces: a deterministic 24-story, 72-fact dataset with complete draft annotations.

- [ ] **Step 1: Write a failing real-dataset distribution test**

```python
def test_real_dataset_has_confirmed_shape():
    data, summary = load_story_dataset(DATASET, ROOT, INDEX)
    self.assertEqual(summary["cases"], 24)
    self.assertEqual(summary["gold_facts"], 72)
    self.assertEqual(summary["verdict_counts"], {"一致": 24, "矛盾": 24, "不确定": 24})
    self.assertEqual(summary["group_counts"], {"zero_conflict": 8, "one_conflict": 8, "multi_conflict": 8})
```

- [ ] **Step 2: Run the test and verify RED because the 24-story file is absent**

- [ ] **Step 3: Migrate the current four stories to the expanded schema**

Preserve story text and existing gold meaning. Add exhaustive `ignored_events`, `non_events`, `gold_findings`, dimensions, normalized facts, relevant lore IDs, and minimum evidence sets. Preserve their IDs for regression comparison.

- [ ] **Step 4: Author twenty additional stories and annotations**

The current four cases contain three zero-conflict stories, one one-conflict story, and no multi-conflict story. Add exactly five zero-conflict, seven one-conflict, and eight multi-conflict stories. Verify final group totals mechanically rather than relying on prose. The final facts must follow the exact 24/24/24 verdict matrix in the spec.

- [ ] **Step 5: Audit story construction against the eight conflict dimensions**

Each one-conflict story owns one unique dimension. Every multi-conflict story has exactly two gold conflict findings from different dimensions. Add routine and referential distractors without using unannotated checkable facts.

- [ ] **Step 6: Switch formal CLI dataset path and defaults to 24 stories**

Set the formal story dataset constant to `evaluation/story_benchmark_24_v2.json`. `validate` must report 24 cases, 72 facts, verdict counts, group counts, and annotation status.

- [ ] **Step 7: Run real-dataset validation and full non-model tests**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 -m unittest tests.test_agent_pipeline_v2_story_dataset_real tests.test_agent_pipeline_v2_story_dataset`

Expected: all pass with exact counts and hashes.

- [ ] **Step 8: Present the 24-story annotation draft for user review**

Do not mark the dataset user-approved in this task. Produce a compact review document grouped by case with gold facts, ignored events, conflict findings, dimensions, and evidence IDs.

---

### Task 3: Complete Review Ledger and Candidate Matcher

**Files:**
- Create: `agent_pipeline_v2/review.py`
- Create: `tests/test_agent_pipeline_v2_review.py`
- Modify: `agent_pipeline_v2/benchmark_cli.py`

**Interfaces:**
- Produces: `build_review_template(dataset: dict, run_rows: list[dict]) -> dict`
- Produces: `propose_review(dataset: dict, run_rows: list[dict]) -> dict`
- Produces: `validate_review(dataset: dict, run_rows: list[dict], review: dict, *, require_complete: bool) -> dict`
- Consumes: every emitted event, retrieval item, Judge item, and final finding.

- [ ] **Step 1: Write failing tests for complete output disposition**

```python
def test_review_requires_every_emitted_event_and_finding():
    review = reviewed_fixture()
    del review["system_event_labels"]["SL-001"]["E2"]
    with self.assertRaisesRegex(ValueError, "unreviewed event"):
        validate_review(DATASET, RUN_ROWS, review, require_complete=True)
```

- [ ] **Step 2: Verify RED**

Run the focused review tests and confirm the missing implementation fails.

- [ ] **Step 3: Implement pending template generation**

Generate empty gold mappings plus one pending label for every system event and final finding. Store dataset/raw digests and reject drift.

- [ ] **Step 4: Implement deterministic candidate proposals**

Rank candidate events by exact anchor overlap, normalized subject tokens, fact text overlap, and cited source IDs. Rank findings by mapped event IDs and evidence IDs. Proposals never change `status` to `reviewed` and never count as final labels.

- [ ] **Step 5: Implement strict reviewed-ledger validation**

Require every system event label to be one of `valid_checkable`, `overselected`, `hallucinated`, or `duplicate`; every finding label to be one of `true_positive`, `false_positive`, `duplicate`, or `unsupported`; mappings to reference existing IDs; duplicate rows to identify the original output; and reviewer notes for hallucinated/unsupported outputs.

- [ ] **Step 6: Integrate the review template into `run-stories`**

Replace the old mapping-only template while retaining a migration error that clearly explains why old review files cannot be scored under the expanded protocol.

- [ ] **Step 7: Run focused tests and update `LOG.md`**

---

### Task 4: Extraction Precision and Hallucination Scoring

**Files:**
- Create: `agent_pipeline_v2/scoring/extraction.py`
- Create: `agent_pipeline_v2/scoring/__init__.py`
- Create: `tests/test_agent_pipeline_v2_extraction_metrics.py`

**Interfaces:**
- Produces: `score_extraction(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict`
- Consumes: reviewed event mappings and `system_event_labels`.

- [ ] **Step 1: Write failing denominator tests**

```python
def test_extra_events_reduce_precision_and_split_hallucination_from_overselection():
    result = score_extraction(CASES, RUNS, REVIEW)
    self.assertEqual(result["recall"], {"hits": 1, "total": 2, "value": 0.5})
    self.assertEqual(result["precision"], {"hits": 1, "total": 3, "value": 1/3})
    self.assertEqual(result["hallucination_rate"]["count"], 1)
    self.assertEqual(result["overselection_rate"]["count"], 1)
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement fact recall and output-based precision**

Count every emitted event exactly once. Valid duplicates do not increase true positives. Hallucinated and overselected outputs remain distinct error classes and both stay in the precision denominator.

- [ ] **Step 4: Add per-case, per-group, and per-dimension breakdowns**

- [ ] **Step 5: Test zero-event and zero-gold cases**

Zero denominators must return `None`; a routine-only story with zero emitted events is a successful clean case.

- [ ] **Step 6: Run focused tests and log the checkpoint**

---

### Task 5: Retrieval Recall, Precision, Noise, and MRR

**Files:**
- Create: `agent_pipeline_v2/scoring/retrieval.py`
- Create: `tests/test_agent_pipeline_v2_retrieval_metrics.py`

**Interfaces:**
- Produces: `score_retrieval(dataset_cases: list[dict], run_rows: list[dict], review: dict, *, k: int = 5) -> dict`

- [ ] **Step 1: Write failing tests for alternative evidence sets and rank**

```python
def test_recall_uses_minimum_sets_and_mrr_uses_first_relevant_rank():
    score = score_retrieval(CASES, RUNS, REVIEW, k=5)
    self.assertEqual(score["recall_at_k"]["hits"], 1)
    self.assertEqual(score["mrr"]["value"], 0.5)
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement gold-level Recall@5 and MRR**

Use the best valid mapped event for gold-level recall and reciprocal rank. Facts without a required sufficient evidence set do not enter the Recall@5 denominator.

- [ ] **Step 4: Implement event-level Precision@5 and Noise Rate**

Judge returned chunks against `relevant_lore_ids` plus reviewed relevance overrides. Report numerator and denominator, not only rates.

- [ ] **Step 5: Add per-dimension breakdown and tests for missing retrieval rows**

- [ ] **Step 6: Run tests and update `LOG.md`**

---

### Task 6: Judge Classification and Unsupported Reasoning Metrics

**Files:**
- Create: `agent_pipeline_v2/scoring/judge.py`
- Create: `tests/test_agent_pipeline_v2_judge_metrics_expanded.py`
- Modify: `agent_pipeline_v2/benchmark.py`

**Interfaces:**
- Produces: `score_pipeline_judge(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict`
- Produces: `classification_metrics(rows: list[dict], labels: tuple[str, ...]) -> dict`
- Extends: `score_judge_results(results: list[dict]) -> dict` with macro-F1 and full per-class precision/recall/F1.

- [ ] **Step 1: Write failing confusion-matrix and macro-F1 tests**

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement deterministic three-class metrics**

Report overall accuracy, confusion matrix, macro-F1, and per-class precision/recall/F1 with explicit denominators. Execution errors appear as incorrect predictions in oracle Judge accuracy.

- [ ] **Step 4: Implement pipeline-conditioned and strict-evidence Judge views**

The pipeline-conditioned view includes extracted facts that reached Judge. The strict view includes only facts whose minimum sufficient evidence was retrieved. Keep extraction and retrieval misses out of the Judge denominator and expose their counts separately.

- [ ] **Step 5: Score unsupported reasoning from reviewed labels**

- [ ] **Step 6: Run focused and legacy benchmark tests**

---

### Task 7: Finding-Level End-to-End Detection Metrics

**Files:**
- Create: `agent_pipeline_v2/scoring/detection.py`
- Create: `tests/test_agent_pipeline_v2_detection_metrics.py`
- Modify: `agent_pipeline_v2/benchmark.py`

**Interfaces:**
- Produces: `score_detection(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict`

- [ ] **Step 1: Write failing TP/FP/FN and duplicate tests**

```python
def test_duplicate_conflict_gets_one_tp_and_one_duplicate():
    score = score_detection(CASES, RUNS, REVIEW)
    self.assertEqual(score["conflict"], {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
    self.assertEqual(score["duplicate_finding_rate"]["count"], 1)
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement one-credit-per-gold finding matching**

Unmatched contradiction findings are FP. Unsupported and wrong-scope conflict findings are FP. Duplicate findings do not become additional TP and are reported separately.

- [ ] **Step 4: Implement omission and story-level metrics**

Add report omission, unsupported report claim, exact conflict-set match, and any-false-alarm rates.

- [ ] **Step 5: Test multi-fact merged findings and partial/error stories**

- [ ] **Step 6: Run tests and log the checkpoint**

---

### Task 8: Earliest-Stage Error Attribution

**Files:**
- Create: `agent_pipeline_v2/attribution.py`
- Create: `tests/test_agent_pipeline_v2_attribution.py`

**Interfaces:**
- Produces: `build_error_attribution(dataset: dict, run_rows: list[dict], review: dict, quality: dict) -> dict`
- Produces one primary error record per product-visible failure.

- [ ] **Step 1: Write failing precedence tests**

```python
def test_extractor_hallucination_remains_primary_when_judge_accepts_it():
    errors = build_error_attribution(DATASET, RUNS, REVIEW, QUALITY)["errors"]
    self.assertEqual(errors[0]["error_type"], "EXTRACTION_HALLUCINATION")
    self.assertEqual(errors[0]["primary_stage"], "extractor")
    self.assertIn("judge", errors[0]["contributing_stages"])
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement the ten fixed error types**

Attach case/gold/event/retrieval/Judge/finding IDs and exact supporting evidence. Generate stable error IDs ordered by case, story position, and stage.

- [ ] **Step 4: Implement earliest-stage precedence and de-duplication**

An upstream miss owns the primary error. Retrieval and Judge effects become contributing stages unless they independently create a second product-visible failure.

- [ ] **Step 5: Separate automatic suggestion from reviewed classification**

Write both fields and fail scoring if a required reviewed disposition is absent.

- [ ] **Step 6: Run focused tests and update `LOG.md`**

---

### Task 9: Fixed Batch-8 Three-Repeat Benchmark

**Files:**
- Modify: `agent_pipeline_v2/benchmark_cli.py`
- Modify: `agent_pipeline_v2/benchmark.py`
- Modify: `agent_pipeline_v2/benchmark_worker.py`
- Modify: `tests/test_agent_pipeline_v2_benchmark.py`

**Interfaces:**
- Produces: batch-8-only CLI behavior and repeat distribution statistics.
- Preserves: historical aggregate readers for batch 1/2/4/8 files.

- [ ] **Step 1: Write failing tests for rejecting formal batch sizes other than 8**

- [ ] **Step 2: Write failing tests for median/min/max/std aggregation**

Use population standard deviation and return `0.0` for a single numeric run. Preserve every raw run.

- [ ] **Step 3: Verify RED**

- [ ] **Step 4: Set formal defaults to batch 8 and repeats 3**

Remove new-run batch comparison output. Keep a compatibility parser for historical result inspection, not execution.

- [ ] **Step 5: Add TTFT, decode speed, token percentiles, and padding summaries**

Percentiles use a documented deterministic nearest-rank or interpolation method tested with fixed samples.

- [ ] **Step 6: Run benchmark unit tests and non-model CLI validation**

---

### Task 10: Unified Quality, Attribution, and Markdown Reports

**Files:**
- Create: `agent_pipeline_v2/summary.py`
- Create: `tests/test_agent_pipeline_v2_summary.py`
- Modify: `agent_pipeline_v2/benchmark_cli.py`
- Modify: `agent_pipeline_v2/README.md`
- Modify: `docs/benchmarks/AGENT_PIPELINE_V2_BENCHMARK.md`

**Interfaces:**
- Produces: `build_benchmark_summary(story_quality: dict, attribution: dict, judge_report: dict, run_status: dict) -> dict`
- Produces: `render_benchmark_markdown(summary: dict) -> str`

- [ ] **Step 1: Write failing JSON/Markdown parity tests**

Assert that every displayed rounded number originates from the same JSON field and that `None` renders as `N/A`.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Build the unified JSON summary**

Include headline Conflict F1 and Extraction Hallucination Rate; extraction, retrieval, Judge, and detection tables; error counts; per-dimension results; completion/failure counts; and batch-8 performance distributions.

- [ ] **Step 4: Render the Markdown report automatically**

Write the headline quality table, stage table, taxonomy table, dimension table, reliability table, and performance table after scoring. Print the summary path in the terminal.

- [ ] **Step 5: Add manifest digests**

Bind dataset, raw story output, review, Judge fixture/runs, prompts, relevant source, quality, attribution, and summaries.

- [ ] **Step 6: Run summary tests and update documentation**

---

### Task 11: Full Verification and User-Run Handoff

**Files:**
- Modify: `LOG.md`
- Generated by user run: `agent_pipeline_v2/reports/<new-run>/...`

**Interfaces:**
- Consumes all preceding tasks.
- Produces a validated command and review workflow for the real CUDA benchmark.

- [ ] **Step 1: Run all unit tests**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 -m unittest discover -s tests`

Expected: zero failures and zero errors.

- [ ] **Step 2: Run syntax compilation with an external pycache**

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'worldcheck-compile-cache'
& ".\.venv\Scripts\python.exe" -X utf8 -m compileall -q -f agent_pipeline_v2 tests
```

- [ ] **Step 3: Validate both historical freezes**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline.verify_freeze`

Expected: `status=ok`, `changed=[]`, `archive_ok=true` for both freezes.

- [ ] **Step 4: Validate the real dataset without loading Qwen**

Run: `& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" validate`

Expected: 24 stories, 72 facts, 24 per verdict, 48 Judge fixtures, batch size 8, three Judge repeats.

- [ ] **Step 5: Give the user the formal CUDA command**

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\agent_pipeline_v2_benchmark.py" run-all --device cuda --judge-batch-size 8 --repeats 3 --top-k 5
```

- [ ] **Step 6: Review the generated 24-story ledger**

Confirm every gold mapping, emitted-event disposition, finding mapping, reasoning-support label, and retrieval relevance override. Mark `status=reviewed` only when complete.

- [ ] **Step 7: Run scoring and inspect attribution**

Run `score-stories` on the reviewed directory. Verify JSON/Markdown parity and manually inspect every hallucination, FP, FN, unsupported reason, duplicate, and report omission.

- [ ] **Step 8: Record the final measured baseline in `LOG.md`**

State sample limits, annotation status, all denominators, performance distributions, and the highest-frequency primary error types. Do not freeze the expanded result until the user explicitly requests it.
