# Agent Pipeline v2 Benchmark Expansion Design

## Goal

Expand the existing Agent Pipeline v2 benchmark from a four-story pilot into a reviewed 24-story detection benchmark. Fix Judge inference at batch size 8, measure false positives and hallucinations, attribute errors to the earliest responsible pipeline stage, and preserve the existing v1 freezes and historical v2 runs.

This remains the Agent Pipeline v2 benchmark. It does not create a v3 runtime or replace historical reports.

## Fixed experimental configuration

- Keep `Qwen/Qwen3-4B-Instruct-2507`, its pinned revision, NF4 CUDA loading, the current Judge prompt, BGE model, Top-5 retrieval, lore files, and index version fixed.
- Set the supported benchmark Judge batch size to 8. Historical batch 1/2/4/8 results remain immutable evidence.
- Run the 24-story track once per formal evaluation.
- Run the existing 48-item oracle-evidence Judge fixture three times in clean subprocesses and aggregate the median, minimum, maximum, standard deviation, and all raw runs.
- Do not introduce length bucketing in this milestone. Preserve fixed item order so padding measurements describe the current batching implementation.
- Retain the existing four stories and add twenty stories.

## Benchmark tracks

### Story detection track

The story track evaluates the complete path:

```text
story -> extractor -> retrieval -> Judge -> report
```

Its primary unit is a reviewed story fact or final conflict finding. It measures extraction quality, retrieval quality, pipeline-conditioned Judge quality, final conflict detection, hallucination, false alarms, duplication, and report omissions.

### Oracle-evidence Judge track

The existing 48-item Judge fixture remains a separate diagnostic track. Every item has a fixed fact, fixed lore, and expected verdict. It measures Judge classification and batch-8 performance without extraction or live retrieval as confounders.

The two tracks must never be merged into one accuracy denominator.

## Story dataset composition

The dataset contains 24 stories and 72 checkable gold facts.

| Group | Stories | Consistent | Contradiction | Uncertain | Total facts |
|---|---:|---:|---:|---:|---:|
| zero_conflict | 8 | 12 | 0 | 12 | 24 |
| one_conflict | 8 | 8 | 8 | 8 | 24 |
| multi_conflict | 8 | 4 | 16 | 4 | 24 |
| total | 24 | 24 | 24 | 24 | 72 |

The zero-conflict group contains two routine-only stories, three stories whose checkable facts are supported by lore, and three stories containing new or currently unsupported concepts. The one-conflict group contains one story for each dimension: time, character knowledge, space, character relation, physical rule, world rule, causality, and identity. Each multi-conflict story contains exactly two gold conflicts from different dimensions plus supported, uncertain, referential, omitted-subject, and routine distractors according to the distribution above.

All stories and lore are explicitly test material and do not represent formal canon.

## Dataset schema

The expanded dataset keeps corpus snapshot hashes and stores exhaustive story annotations. Each case has this conceptual shape:

```json
{
  "id": "SL-001",
  "title": "...",
  "group": "zero_conflict | one_conflict | multi_conflict",
  "dimensions": ["time"],
  "story": "...",
  "character_count": 240,
  "gold_facts": [],
  "ignored_events": [],
  "non_events": [],
  "gold_findings": [],
  "annotation_status": "assistant_authored_draft_pending_human_review"
}
```

Each `gold_fact` contains:

```json
{
  "id": "G1",
  "subject": "雷",
  "predicate": "拥有",
  "object": "两颗心脏",
  "normalized_fact": "雷拥有两颗心脏",
  "type": "character_attribute",
  "dimension": "physical_rule",
  "source_anchors": [{"text": "他有两颗心脏。", "start": 21, "end": 28}],
  "context_anchors": [{"text": "雷独自坐在桌边，", "start": 0, "end": 8}],
  "expected_verdict": "一致",
  "relevant_lore_ids": ["099d1bbdbc348eb3c332"],
  "minimum_evidence_sets": [["099d1bbdbc348eb3c332"]],
  "reportable": false
}
```

`relevant_lore_ids` exhaustively lists lore chunks that materially support or contradict the fact. `minimum_evidence_sets` lists alternative minimal sufficient evidence combinations. Uncertain facts may have relevant contextual lore but no sufficient evidence set.

Each `ignored_event` contains an exact source anchor, normalized proposition, and reason code `routine` or `process_detail`. Each `non_event` contains an exact source anchor and reason. These annotations must cover all intentional distractors required to distinguish unsupported hallucination from over-selection.

Each `gold_finding` references one or more `gold_fact_ids`, identifies the expected final finding type, dimension, sufficient evidence sets, and whether multiple facts should be merged into one author-facing finding.

## Annotation lifecycle

1. The assistant authors all 24 stories and exhaustive draft annotations.
2. Dataset validation checks exact anchors, offsets, IDs, distributions, lore references, evidence-set membership, and corpus hashes without loading Qwen.
3. The benchmark produces system events, retrieval results, Judge items, and report findings.
4. An automatic matcher proposes gold-to-event and gold-finding-to-report mappings using anchors, subjects, normalized propositions, and evidence IDs.
5. A human reviewer confirms, edits, or rejects every mapping and classifies every unmatched system output.
6. Only a review file with `status: reviewed` can be scored.
7. The dataset remains draft until the user approves the complete 24-story annotations. Formal report metadata records the dataset and review digests.

The matcher may propose candidates but cannot award final credit without reviewed mappings.

## Review schema

`story_review.json` stores:

```json
{
  "status": "pending_human_review | reviewed",
  "event_mapping": {"SL-001": {"G1": ["E1"]}},
  "finding_mapping": {"SL-001": {"F1": ["R1"]}},
  "system_event_labels": {
    "SL-001": {
      "E2": {
        "label": "valid_checkable | overselected | hallucinated | duplicate",
        "gold_ids": [],
        "note": "..."
      }
    }
  },
  "system_finding_labels": {
    "SL-001": {
      "R2": {
        "label": "true_positive | false_positive | duplicate | unsupported",
        "gold_finding_ids": [],
        "note": "..."
      }
    }
  },
  "retrieval_relevance_overrides": {},
  "reasoning_support_labels": {}
}
```

Every emitted event and final finding must receive a reviewed disposition. This prevents unreviewed extra output from disappearing from precision denominators.

## Matching rules

- A gold fact may map to multiple system events when the extractor splits one proposition.
- One system event may map to multiple gold facts only when its cited source spans and normalized proposition explicitly cover every mapped fact.
- Correct verdict alone is insufficient for a match.
- Subject, time, side, location, modality, negation, quantity, and relevant conditions must agree.
- An inferred summary cannot replace explicit facts it does not source.
- Routine actions cannot be mapped to gold to improve precision.
- Semantically similar facts with different subjects or scopes remain unmatched.
- Duplicate outputs map to the same gold but only one receives detection credit; the others count toward duplication metrics.

## Metric definitions

Undefined ratios use JSON `null` and Markdown `N/A`; they are never coerced to zero.

### Extraction metrics

Let a reviewed emitted event be `valid_checkable`, `overselected`, `hallucinated`, or `duplicate`.

```text
Extraction Recall = matched gold facts / all gold facts
Extraction Precision = valid non-duplicate emitted events / all emitted events
Extraction Hallucination Rate = hallucinated emitted events / all emitted events
Extraction Overselection Rate = overselected emitted events / all emitted events
Extraction Duplicate Rate = duplicate emitted events / all emitted events
```

An event is hallucinated when its proposition is not supported by its cited story spans and necessary context. An event is overselected when the proposition exists in the story but is annotated as routine or process detail rather than worldbuilding-relevant.

### Retrieval metrics

Retrieval metrics apply only to matched gold facts with reviewed system events.

```text
Recall@5 = gold facts whose Top-5 contains a minimum sufficient evidence set / gold facts requiring sufficient lore evidence
Precision@5 = relevant returned lore chunks / all returned lore chunks for scored gold facts
Noise Rate@5 = irrelevant returned lore chunks / all returned lore chunks for scored gold facts
MRR = mean reciprocal rank of the first relevant lore chunk
```

If multiple emitted events map to one gold, use the best valid retrieval result for Recall@5 and MRR, but count every returned chunk once in event-level precision/noise diagnostics. Reports expose both gold-level and event-level denominators.

### Judge metrics

Report overall accuracy, macro-F1, per-class precision/recall/F1, and a three-class confusion matrix for consistent, contradiction, and uncertain. Pipeline Judge metrics include only gold facts that were extracted and reached Judge, with a second strict metric restricted to facts whose minimum evidence set was retrieved. Missing upstream facts remain extraction/retrieval failures rather than being silently added to the Judge denominator.

```text
Unsupported Reasoning Rate = reviewed unsupported Judge reasons / reviewed Judge answers
```

The oracle-evidence 48-item track reports Judge accuracy and macro-F1 over all 48 items; parse failures and execution failures are incorrect.

### End-to-end conflict detection

The primary quality metric is finding-level conflict F1.

```text
TP = one reviewed system conflict matched to one gold conflict finding
FP = an unmatched, unsupported, or wrong-scope system conflict finding
FN = a gold conflict finding with no matched system conflict
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2TP / (2TP + FP + FN)
```

Additional metrics:

```text
Duplicate Finding Rate = duplicate conflict findings / all reported conflict findings
Report Omission Rate = correctly judged reportable gold facts absent from final report / correctly judged reportable gold facts
Unsupported Report Claim Rate = unsupported reviewed report claims / all report claims
Story Exact Match = stories with exactly the correct conflict set / all stories
Story Any-False-Alarm Rate = stories with at least one FP / all stories
```

## Error attribution

Every reviewed error creates one record in `error_attribution.json`. Supported error types are:

- `EXTRACTION_MISS`
- `EXTRACTION_HALLUCINATION`
- `EXTRACTION_OVERSELECT`
- `RETRIEVAL_MISS`
- `RETRIEVAL_NOISE`
- `JUDGE_FALSE_POSITIVE`
- `JUDGE_FALSE_NEGATIVE`
- `UNSUPPORTED_REASONING`
- `DUPLICATE_FINDING`
- `REPORT_OMISSION`

The automatic system proposes the earliest stage that introduced the error. The reviewed record stores `primary_stage`, optional `contributing_stages`, exact story spans, event IDs, retrieval chunk IDs, Judge item IDs, finding IDs, an automatic explanation, review status, and reviewer note.

One product-visible error receives one primary attribution. Downstream consequences are attached as contributing stages and do not inflate the total error count. For example, an extractor hallucination that retrieves coincidentally related lore and is accepted by Judge remains primarily an extraction hallucination.

## Fixed batch-8 performance protocol

The oracle Judge worker runs three clean processes. Each process:

1. loads the pinned model;
2. records load time and post-load CUDA allocation;
3. performs one batch-8 warm-up;
4. resets CUDA peak statistics;
5. processes all 48 fixed items in fixed order at batch 8;
6. writes raw item, batch, timing, token, retry, and memory records;
7. exits.

Aggregate metrics include median, minimum, maximum, standard deviation, and each raw value for model load time, peak allocated/reserved VRAM, incremental peak, total Judge time, GPU generation time, facts/s, padding ratio, generated tokens, parse failures, and retries. TTFT reports P50/P95. Decode speed is total decoded tokens divided by total decode seconds. Input tokens report mean, P50, P95, and maximum.

No batch-size comparison table is generated for new runs. Historical batch scaling tables remain unchanged.

## CLI behavior

The current entry point remains `agent_pipeline_v2_benchmark.py`.

- `validate` validates the 24-story dataset, 72-gold distribution, lore/index snapshot, 48 Judge fixtures, and supported batch size 8 without loading Qwen.
- `run-stories` runs the 24 stories once and writes a pending review template.
- `score-stories` requires a complete reviewed file and writes quality plus attribution outputs.
- `run-judge` accepts repeats but uses batch size 8 only; the formal default is three repeats.
- `run-all` runs stories once and Judge three times, then writes explicit per-track status.

Passing another batch size is an explicit validation error. Historical result readers still accept batch 1/2/4/8 files.

## Outputs

Each formal result directory contains:

```text
story_dataset.json
annotation_manifest.json
stories_raw.jsonl
story_review.json
story_quality.json
error_attribution.json
judge_fixture.json
judge_repeat_1.json
judge_repeat_2.json
judge_repeat_3.json
judge_report.json
benchmark_summary.json
benchmark_summary.md
run_status.json
source_manifest.json
```

`benchmark_summary.md` contains a headline quality table, extraction/retrieval/Judge breakdown, end-to-end detection table, error taxonomy counts, per-dimension results, completion/retry failures, and fixed batch-8 performance statistics. `benchmark_summary.json` contains the same values without rounded display formatting.

Raw outputs remain immutable after scoring. Review, quality, attribution, and summary files carry SHA-256 links to the dataset and raw outputs used.

## Validation and failure handling

- Reject missing, duplicate, or invalid case/gold/finding IDs.
- Verify every source/context/ignored/non-event anchor by exact text and offsets.
- Verify the 24-story and 72-fact class distributions.
- Verify all lore IDs against the pinned index and every minimum evidence set against `relevant_lore_ids`.
- Reject scoring when any emitted event or finding lacks a reviewed disposition.
- Reject scoring when mapped IDs do not exist, mappings violate subject/scope constraints, or review/dataset/raw digests mismatch.
- Preserve `partial` and `error` outputs in denominators where specified; never silently drop failed stories.
- Keep automatic attribution suggestions separate from reviewed attribution.

## Testing strategy

Unit tests cover dataset validation, exact anchor checking, class distributions, event/finding matching, one-to-many and duplicate rules, all metric denominators, undefined ratios, retrieval alternatives, MRR, hallucination versus over-selection, attribution precedence, review completeness, three-repeat aggregation, fixed batch enforcement, summary rendering, digest mismatch, and partial/error cases.

Integration tests use deterministic synthetic reports to exercise the entire scoring path without loading models. A final non-generative validation checks the real 24-story dataset and current index. The real CUDA benchmark is run by the user after unit, compile, and freeze checks pass.

Existing Agent Pipeline v1 and Story-Level benchmark freezes must still report `status=ok`, `changed=[]`, and valid archives after implementation.

## Acceptance criteria

- Dataset validation reports exactly 24 stories and 72 gold facts, with 24 facts per verdict class.
- Every system event and final finding requires a reviewed label before scoring.
- Gold recall, output precision, hallucination, over-selection, retrieval, Judge, and end-to-end finding metrics are all produced with explicit denominators.
- Every reviewed error has one primary stage, optional contributing stages, and traceable evidence.
- Formal Judge runs use batch 8 for three independent repeats and report distribution statistics.
- The Markdown and JSON summaries agree before rounding.
- The original four stories remain regression cases.
- Existing historical results and freezes remain unchanged.
