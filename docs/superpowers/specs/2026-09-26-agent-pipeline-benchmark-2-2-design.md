# Agent Pipeline Benchmark 2.2 Design

## 1. Goal

Benchmark 2.2 compares the frozen Agent Pipeline v2 end to end against a candidate pipeline that adds the two changes validated separately in Benchmark 2.1:

- retrieval changes from Dense Top-5 to Hybrid RRF Top-5;
- Judge changes from frozen Judge v1 to Judge v2.1, where contradiction requires that the story fact and applicable lore cannot both be true.

The benchmark keeps the existing 24-story dataset. Dataset expansion is outside Benchmark 2.2.

## 2. Compared systems

### Baseline

- Extraction: frozen Agent Pipeline v2 one-step checkable-event extractor.
- Retrieval: frozen Dense Top-5 retrieval.
- Judge: frozen Judge v1.
- Judge batch size: 8.
- Device: CUDA for the official run.

### Candidate

- Extraction: the same extractor version and extraction-repair policy used by the Benchmark 2.2 implementation.
- Retrieval: Hybrid RRF Top-5, combining Dense and Chinese BM25.
- Metadata filter: disabled.
- Judge: Judge v2.1 coexistence rule.
- Judge batch size: 8.
- Device: CUDA for the official run.

The candidate must run as a complete story pipeline. It may reuse tested v2.1 components, but it must not manufacture a candidate result by editing stored Baseline output.

## 3. Two evaluation tracks

### End-to-end track

Baseline and Candidate each start from the original story text and execute their complete pipeline. This track measures user-visible quality, total latency, stage latency, execution failures, and peak VRAM.

Because extraction is model-generated, the two end-to-end runs may emit different events. Those differences are part of observed end-to-end behavior.

### Paired attribution track

Both Retrieval/Judge variants consume the same reviewed extraction events. This track isolates the effect of Dense versus Hybrid RRF and Judge v1 versus Judge v2.1.

The paired track uses a two-by-two matrix:

| Retrieval | Judge | Purpose |
|---|---|---|
| Dense | v1 | Paired Baseline |
| Hybrid RRF | v1 | Retrieval-only delta |
| Dense | v2.1 | Judge-only delta |
| Hybrid RRF | v2.1 | Combined Candidate delta |

This matrix prevents an apparent Judge improvement from actually being caused by different evidence, and prevents a retrieval improvement from being hidden by a Judge regression.

## 4. Fixed experimental controls

- Dataset: the existing 24-story, 72-gold-fact fixture used by Agent Pipeline v2/v2.1.
- Lore corpus and embedding index: the same pinned snapshots for all variants.
- LLM: `Qwen/Qwen3-4B-Instruct-2507`, with the project-pinned revision and deterministic generation settings.
- Dense encoder: `BAAI/bge-small-zh-v1.5`, with the project-pinned revision and FP32 embeddings.
- Top-K: 5.
- Judge batch size: 8.
- Candidate metadata filter: off.
- Quality runs: one complete deterministic run per end-to-end system.
- Performance runs: one warm-up followed by three measured repeats; headline values use the median.
- A valid but semantically wrong output is not retried.
- Parse failures, token-budget failures, OOMs, and other execution failures remain in the denominator.

## 5. Reviewed mappings

Scoring requires a review ledger that maps:

- every gold fact to zero or more emitted events;
- every emitted event to its gold fact IDs or to `overselected`;
- every gold reportable conflict to zero or more system findings;
- every system finding to `true_positive`, `false_positive`, or `unsupported`;
- unsupported reasoning and error attribution to the exact case and event.

Mappings may be reused only when the story ID, event text, actors, source IDs, and relevant event signature match exactly. Changed outputs require fresh review. Assistant review is labeled `assistant-reviewed` and must not be described as user or independent human review.

## 6. Quality metrics

### Extraction

`Extraction Miss` is a gold fact with no valid mapped system event.

`Recall_Extraction = extracted gold facts / all gold facts`.

The report also records overselected events and extraction hallucination rate, but these do not replace Extraction Miss.

### Retrieval

Retrieval is eligible only when its gold fact was extracted.

`Retrieval Miss` is an eligible fact for which Top-5 does not contain a complete minimum evidence set.

`Recall@5_Retrieval = eligible facts with a complete evidence set / eligible facts requiring lore evidence`.

The report must also retain evidence-level micro Recall@5 and MRR@5 as diagnostics.

### Judge

Judge accuracy is calculated on extracted facts with a reviewed expected verdict. Its confusion matrix uses `consistent`, `contradiction`, and `uncertain`.

`Judge FP` is a non-contradiction gold fact reported as contradiction.

`Judge FN` is a contradiction gold fact not reported as contradiction. The report separates:

- total Judge FN;
- Judge FN with complete gold evidence already retrieved;
- upstream-caused misses where required evidence was absent.

This separation prevents retrieval failures from being counted as pure Judge failures.

### Unsupported reasoning

Unsupported reasoning is counted when the verdict or reason relies on a claim that the cited evidence does not establish, including unsupported identity equivalence, location equivalence, temporal assumptions, causal rules, exclusivity, capability transfer, or scope generalization.

It is reviewed independently from whether the final class happens to match the gold label.

### End-to-end conflict detection

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`

A TP requires a reviewed report finding mapped to a reportable gold conflict. Missing extraction, retrieval failure, Judge failure, report failure, and execution failure can all produce an end-to-end FN.

## 7. First-failure attribution

Each missed reportable gold conflict receives exactly one primary failure attribution, in this order:

1. `execution_failure`
2. `extraction_miss`
3. `retrieval_miss`
4. `judge_fn`
5. `report_miss`

Judge false positives and unsupported reasoning are separate output errors and are not forced into the missed-conflict chain.

The scorer must expose both aggregate counts and case/fact-level records so every headline number is auditable.

## 8. Performance metrics

For each end-to-end system, record:

- model and index load time;
- extraction time;
- query encoding and retrieval time;
- Judge time;
- report time;
- total end-to-end time;
- stories per second and judged facts per second;
- generated and input token counts when available;
- peak CUDA allocated GiB;
- peak CUDA reserved GiB;
- complete/partial/error case counts.

Headline timing values are medians of three measured runs after warm-up. Quality is scored from the designated deterministic quality run, not averaged across performance repeats.

## 9. Architecture and isolation

Create an independent `agent_pipeline_v2_2` package and `agent_pipeline_v2_2_benchmark.py` entry point.

The package contains:

- a Candidate story pipeline composed from the v2 extractor, Hybrid RRF without metadata filtering, Judge v2.1, and the existing deterministic report builder;
- a Baseline adapter that calls frozen v2 behavior without modifying it;
- a paired-matrix runner;
- a review-ledger generator and validator;
- a Benchmark 2.2 scorer with first-failure attribution;
- a report generator for JSON and Markdown comparison tables.

Frozen v2, v2.1, previous datasets, prompts, reports, and freeze manifests must not be modified.

## 10. CLI and outputs

The Benchmark 2.2 CLI provides independently resumable commands:

- `validate`
- `run-end-to-end`
- `run-paired`
- `score`
- `summary`
- `run-all`

Official output is written to a unique directory under `agent_pipeline_v2_2/reports/`. It includes:

- copied dataset and configuration manifest;
- Baseline and Candidate raw JSONL;
- paired-matrix raw results;
- execution summaries;
- pending and reviewed ledgers;
- quality metrics and first-failure attribution;
- performance metrics;
- JSON and Markdown comparison reports;
- SHA256 hashes for data, lore, index, prompts, source files, and outputs.

Intermediate writes are atomic where practical. An interrupted run keeps completed case rows and can resume only when dataset and configuration hashes match.

## 11. Status semantics

- `ok`: all required cases and stages completed and the review ledger is valid.
- `partial`: at least one required case or stage failed, but usable results exist.
- `error`: no valid comparison can be scored or a required invariant is broken.
- `pending_review`: execution completed, but reviewed mappings are not yet available.

Semantic uncertainty is a valid verdict and does not make a run partial.

## 12. Verification and acceptance

Before the official GPU run:

- unit tests cover all metric formulas and zero-denominator behavior;
- attribution tests prove each missed gold conflict receives one primary cause;
- paired-matrix tests prove extraction events are identical across its four cells;
- integration tests prove Baseline uses Dense/v1 and Candidate uses Hybrid/no-metadata/v2.1;
- resume tests reject mismatched data or configuration;
- freeze verification confirms all previous benchmark snapshots are unchanged.

Benchmark 2.2 is complete when:

- both end-to-end variants finish the same 24 stories;
- the four paired cells finish the same reviewed facts;
- all requested quality, attribution, latency, throughput, and VRAM metrics are present;
- raw rows reconcile exactly with aggregate counts;
- the final report clearly separates structural failures, retrieval failures, Judge errors, and model-quality limitations;
- the complete Benchmark 2.2 result is frozen only after review and scoring succeed.

Benchmark 2.2 does not claim generalization beyond the existing 24-story fixture. Dataset expansion remains the next separate benchmark step.
