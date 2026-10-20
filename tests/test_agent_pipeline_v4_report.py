from agent_pipeline_v4.report import compare_extractions, summarize_static, render_table


def _row(case_id, events):
    extraction = {"events": events, "status": "ok", "calls": [], "uncovered_span_ids": []}
    return {"case_id": case_id, "status": "ok", "result": {
        "judge": {"events": events}, "benchmark_stages": {"extraction": extraction}}}


def test_exact_event_reuses_frozen_review_and_new_event_waits_for_review():
    old = {"id": "E1", "actors": ["雷"], "event": "雷胸痛", "source_ids": ["S1"], "context_ids": [], "modality": "observed", "explicit": True, "mental_state": None, "conditions": [], "check_reason": "mechanism"}
    changed = {**old, "id": "E2", "event": "雷头痛"}
    dataset = [{"id": "SL-001", "gold_facts": [{"id": "G1"}]}]
    review = {"system_event_labels": {"candidate:SL-001": {"E1": {"label": "valid_checkable"}}},
              "event_mapping": {"candidate:SL-001": {"G1": ["E1"]}}}
    compared = compare_extractions(dataset, [_row("SL-001", [old])], [_row("SL-001", [old, changed])], review)
    assert compared["pending_events"] == [{"case_id": "SL-001", "event_id": "E2", "event": changed}]
    assert compared["candidate"]["recall"] is None
    assert compared["candidate"]["precision"] is None
    assert compared["baseline"]["recall"] == 1.0
    assert compared["baseline"]["precision"] == 1.0


def test_batch_wall_time_is_not_multiplied_by_story_count():
    calls = [{"call_id": "LLM-000001", "batch_size": 2, "input_tokens": 20, "padded_input_tokens": 4,
              "output_tokens": 10, "prefill_time": 0.2, "decode_time": 0.8, "total_time": 1.0,
              "ttft_ms": 200.0, "peak_allocated": 100, "peak_reserved": 120}]
    rows = [_row("A", []), _row("B", [])]
    summary = summarize_static(rows, calls, wall_seconds=1.4, batch_size=2)
    assert summary["llm_seconds"] == 1.0
    assert summary["wall_seconds"] == 1.4
    assert summary["decode_tokens_per_second"] == 12.5
    assert summary["actual_avg_batch"] == 2.0


def test_resumed_run_has_no_comparable_wall_throughput():
    summary = summarize_static([_row("A", [])], [], wall_seconds=None, batch_size=1)
    assert summary["wall_seconds"] is None
    assert summary["stories_per_second"] is None


def test_overview_displays_oom_size_without_fabricated_metrics():
    rendered = render_table([], {"recall": 0.9, "precision": 0.8},
                            failures=[{"batch_size": 8, "status": "oom", "error": "CUDA out of memory"}])
    assert "| 8 | OOM" in rendered
    assert "CUDA out of memory" in rendered


def test_overview_names_frozen_extractor_performance_reference():
    rendered = render_table([], {"recall": 0.9, "precision": 0.8},
                            baseline_performance={"calls": 103, "output_tokens": 11714,
                                                  "decode_seconds": 970.1, "total_seconds": 1024.1})
    assert "103" in rendered
    assert "11714" in rendered
    assert "970.10" in rendered


def test_frozen_benchmark_three_review_reproduces_its_extraction_scores():
    from agent_pipeline_v4.cli import validate_inputs

    inputs = validate_inputs()
    compared = compare_extractions(inputs["cases"], inputs["source_rows"], inputs["source_rows"], inputs["review"])
    assert compared["quality_status"] == "reviewed"
    assert compared["baseline"]["gold_hits"] == 71
    assert compared["baseline"]["gold_total"] == 72
    assert compared["baseline"]["valid_events"] == 71
    assert compared["baseline"]["emitted_events"] == 80
