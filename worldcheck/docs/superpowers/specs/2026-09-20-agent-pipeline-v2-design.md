# Agent Pipeline v2 Design

## Goal

Build an isolated `agent_pipeline_v2` that extracts only worldbuilding-relevant events in one LLM stage and judges independent fact/lore prompts with true GPU batching. Preserve the frozen v1 implementation and benchmark for comparison.

## Scope

This milestone changes two stages only:

1. Replace v1 understanding followed by filtering with one structured checkable-event extractor.
2. Replace serial per-fact Judge generation with configurable tensor batching and benchmark batch sizes 1, 2, 4, and 8.

Retrieval continues to use the existing BGE model, Top-5 index, and lore. The four-story pilot remains provisional and will be expanded later. Model routing, inference engines, semantic story chunking, and larger models are outside this milestone.

## Isolation

All new runtime code lives under `agent_pipeline_v2/`; the benchmark entry point is `agent_pipeline_v2_benchmark.py`. V1 source and both existing frozen snapshots remain unchanged. Shared stable primitives may be imported from the root encoder, Retriever, model loader, and frozen lore index; v2 does not import v1 understanding, filtering, Judge orchestration, or report schemas.

## Pipeline

```text
story -> source spans -> checkable extractor -> selected events
      -> BGE Top-5 retrieval -> batched Judge -> v2 report
```

The extractor uses one semantic stage. Long inputs may require several bounded windows, but no second LLM filtering stage is allowed.

## Extractor contract

Each source span must have exactly one audited disposition:

- referenced by one or more `events`;
- listed in `ignored_spans` with `routine` or `process_detail`;
- listed in `non_event_span_ids` because it has no independent proposition.

An event contains `id`, `actors`, `event`, `mental_state`, `explicit`, `modality`, `conditions`, `source_ids`, `context_ids`, and `check_reason`. Allowed reasons are `mechanism`, `knowledge_relation`, `state_time_space`, and `consequence_support`. The validator rejects missing coverage, overlapping dispositions, invented span IDs, duplicate IDs, invalid source/context roles, and empty facts. Invalid wire output may be retried once; valid rows from a mixed response are not silently discarded.

## Retrieval contract

V2 retrieval accepts the extractor report directly. Each selected event receives up to five existing lore chunks. It preserves chunk ID, original text, file, line range, heading path, and cosine score. Retrieval errors remain attached to their event.

## Batched Judge

Each event/lore pair remains a separate chat prompt. `BatchedGenerator` applies the chat template independently, uses decoder-only left padding, creates `[batch, max_sequence]` tensors with attention masks, and calls `model.generate()` once per micro-batch. It decodes and validates each row independently.

Valid rows survive mixed batches. Invalid JSON or schema rows are gathered into one retry batch with row-specific format feedback. A second failure remains an error and counts as incorrect. No prompt packing is used. An OOM records the failed configuration, clears CUDA state, and exits that worker without lowering the requested batch size.

## Judge fixture set

Create a deterministic 48-claim fixture from `evaluation/world_benchmark_160_v1.json`: 16 consistent, 16 contradiction, and 16 uncertain. Selection favors distinct families, dimensions, input lengths, and records whose `prior_development_semantic_overlap` is false.

Each fixture stores one atomic fact, a fixed ordered set of up to five lore chunks, and the expected verdict. Consistent and contradiction items must include at least one annotated canonical evidence quote. Uncertain items receive plausible retrieved lore that neither directly supports nor directly contradicts the fact. Fixture generation is outside timed Judge execution; the materialized JSONL and its hashes are benchmark inputs.

## Benchmark protocol

The four-story pilot measures v2 extraction, retrieval, Judge, and end-to-end quality for comparison with v1. The fixed 48-item Judge fixture separately measures batch sizes 1, 2, 4, and 8.

Each batch size runs in a clean subprocess:

1. load the pinned Qwen model;
2. record load time and post-load CUDA allocation;
3. run one complete warm-up batch;
4. reset peak memory statistics;
5. process all 48 fixtures in fixed order;
6. write item results and aggregate metrics;
7. terminate the worker.

One measured pass is the default; `--repeats` permits later repeated runs. Total Judge time includes tokenization, padding, generation, decoding, validation, and retries. It excludes model loading, fixture construction, and retrieval. GPU generation time is also reported separately.

Metrics are peak allocated/reserved VRAM, incremental peak over loaded model, total Judge time, GPU generation time, facts/s, overall and per-class Judge accuracy, confusion matrix, parse failure rate, retry rate, useful/padded input tokens, padding ratio, generated tokens, and main/retry batch counts. Accuracy always uses all 48 facts; invalid output and execution failures are incorrect.

## Outputs

Each run creates a unique directory under `agent_pipeline_v2/reports/` containing the copied dataset, extracted story results, fixed Judge fixtures, one JSON file per batch size, quality metrics, a Markdown comparison table, raw records, and a hash manifest. Dataset and results remain marked provisional.

## Failure handling

Empty input, invalid schemas, uncovered spans, missing indexes, fixture hash drift, mismatched model/prompt versions, and corrupt reports produce explicit errors. Partial row failure never erases valid sibling rows. The CLI never silently truncates source text, lore, or prompts.

## Verification

Tests cover extractor coverage and routing, rejection of overlap and missing spans, retrieval adaptation, left padding and output-to-row alignment, partial retry, OOM reporting, metric definitions, deterministic balanced fixture selection, subprocess aggregation, and v1 freeze integrity. CUDA execution is a separate smoke/benchmark step; unit tests use a deterministic fake batched backend.

