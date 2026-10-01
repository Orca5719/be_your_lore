from __future__ import annotations

import pytest

from agent_pipeline_v3_2.extractor_audit import (
    CATEGORIES,
    aggregate_audit,
    build_audit_rows,
    classify_failure,
)


@pytest.mark.parametrize(
    ("error", "error_type", "raw", "expected"),
    [
        ("Expecting ':' delimiter", "JSONDecodeError", "{bad", "JSON_PARSE_ERROR"),
        ("生成达到输出上限且未结束", "ValueError", "{...", "TRUNCATED_OUTPUT"),
        ("输出必须且只能包含events、ignored_spans、non_event_span_ids", "ValueError", "{}", "TOP_LEVEL_SCHEMA_ERROR"),
        ("事件字段不符合v2 wire协议", "ValueError", "{}", "EVENT_FIELD_ERROR"),
        ("non_event_span_ids只能引用当前target_spans", "ValueError", "{}", "INVALID_REFERENCE_OR_DISPOSITION"),
        ("核对原因无效", "ValueError", "{}", "INVALID_ENUM_OR_TYPE"),
        ("存在未覆盖的target_spans：S1", "ValueError", "{}", "MISSING_TARGET_COVERAGE"),
        ("unknown", "ValueError", "", "EMPTY_OUTPUT"),
        ("unknown", "ValueError", "{}", "OTHER"),
    ],
)
def test_failure_taxonomy(error, error_type, raw, expected):
    assert classify_failure(error=error, error_type=error_type, raw_output=raw) == expected
    assert expected in CATEGORIES


def _call(call_id, status, tokens, seconds):
    return {
        "call_id": call_id,
        "component": "extractor",
        "status": "ok",
        "input_tokens": 100,
        "padded_input_tokens": 0,
        "output_tokens": tokens,
        "prefill_time": 1.0,
        "decode_time": seconds - 1.0,
        "total_time": seconds,
        "batch_size": 1,
        "peak_allocated": 10,
        "peak_reserved": 20,
        "is_retry": call_id != "LLM-000001",
    }


def test_cost_classes_are_exclusive_and_totals_close():
    stories = [{
        "case_id": "SL-001",
        "result": {"benchmark_stages": {"extraction": {"calls": [{
            "window_id": 1,
            "target_ids": ["S1"],
            "attempts": [
                {"attempt": 1, "status": "error", "error": "存在未覆盖的target_spans：S1", "error_type": "ValueError", "raw_output": "{}", "timing": {"call_id": "LLM-000001"}},
                {"attempt": 2, "purpose": "recover_missing_targets", "status": "error", "error": "核对原因无效", "error_type": "ValueError", "raw_output": "{}", "timing": {"call_id": "LLM-000002"}},
                {"attempt": 2, "status": "ok", "raw_output": "{}", "timing": {"call_id": "LLM-000003"}},
            ],
        }]}}},
    }]
    calls = [
        _call("LLM-000001", "ok", 20, 3.0),
        _call("LLM-000002", "ok", 10, 2.0),
        _call("LLM-000003", "ok", 5, 1.5),
    ]
    rows = build_audit_rows(stories, calls)
    assert [row["cost_class"] for row in rows] == ["pure_waste", "pure_waste", "effective_cost"]
    assert [row["outcome"] for row in rows] == ["initial_failure", "recovery_failure", "format_retry_success"]
    assert rows[0]["recovered_by_later"] is True
    assert rows[1]["recovered_by_later"] is True
    report = aggregate_audit(rows)
    assert report["totals"]["calls"] == 3
    assert report["totals"]["output_tokens"] == 35
    assert report["cost_classes"]["pure_waste"]["output_tokens"] == 30
    assert report["cost_classes"]["effective_cost"]["total_time"] == 1.5
    assert report["closure"]["tokens_closed"] is True
    assert report["closure"]["time_closed"] is True


def test_successful_recovery_is_recovery_cost():
    stories = [{
        "case_id": "SL-001",
        "result": {"benchmark_stages": {"extraction": {"calls": [{
            "window_id": 1,
            "target_ids": ["S1"],
            "attempts": [{"attempt": 2, "purpose": "recover_missing_targets", "status": "ok", "raw_output": "{}", "timing": {"call_id": "LLM-000001"}}],
        }]}}},
    }]
    rows = build_audit_rows(stories, [_call("LLM-000001", "ok", 7, 2.0)])
    assert rows[0]["cost_class"] == "recovery_cost"
    assert rows[0]["outcome"] == "recovery_success"
