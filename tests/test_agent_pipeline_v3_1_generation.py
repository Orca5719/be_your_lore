import json

import pytest

from agent_pipeline_v3_1.generation import (
    analyze_generation,
    project_extractor_payload,
    project_judge_payload,
)


def count_chars(text: str) -> int:
    return len(text)


def extractor_call(call_id="LLM-1", output_tokens=200, raw=None, status="ok"):
    raw = raw or json.dumps({
        "events": [{
            "actors": ["雷"], "event": "雷拥有两颗心脏", "mental_state": None,
            "explicit": True, "modality": "observed", "conditions": [],
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
        "non_event_span_ids": ["S3"],
    }, ensure_ascii=False)
    return (
        {
            "call_id": call_id, "component": "extractor", "output_tokens": output_tokens,
            "decode_time": 10.0, "total_time": 11.0, "input_tokens": 100,
        },
        {"attempt": 1, "status": status, "raw_output": raw, "timing": {"call_id": call_id}},
    )


def judge_raw(verdict="contradiction", reason="这是很长的理由" * 20):
    return json.dumps({
        "verdict": verdict,
        "citations": [{"evidence_id": "L1", "quote": "设定原文"}],
        "reason": reason,
        "assessment": {"same_subject": True, "relation": "direct_conflict"},
    }, ensure_ascii=False)


def test_extractor_projection_preserves_event_and_coverage_identities():
    _, attempt = extractor_call()
    payload = json.loads(attempt["raw_output"])
    projected = project_extractor_payload(payload)
    assert projected["events"] == payload["events"]
    assert projected["ignored_span_ids"] == ["S2"]
    assert projected["non_event_span_ids"] == ["S3"]
    assert '"reason":' not in json.dumps(projected, ensure_ascii=False)


def test_judge_projection_keeps_verdict_and_only_conflict_evidence():
    contradiction = project_judge_payload(json.loads(judge_raw("contradiction")))
    consistent = project_judge_payload(json.loads(judge_raw("consistent")))
    assert contradiction["verdict"] == "contradiction"
    assert contradiction["citation_ids"] == ["L1"]
    assert len(contradiction["reason"]) <= 80
    assert consistent == {"verdict": "consistent"}


def test_analysis_separates_real_tokens_estimates_residual_and_projection():
    call, attempt = extractor_call()
    stories = [{
        "case_id": "SL-001",
        "result": {"benchmark_stages": {"extraction": {"events": [{"id": "E1"}], "calls": [{"attempts": [attempt]}]}, "judge": {"batch_reports": []}}},
    }]
    result = analyze_generation(stories, [call], count_chars)
    extractor = result["components"]["extractor"]
    assert extractor["actual_output_tokens"] == 200
    assert extractor["field_estimated_tokens"]["events"] > 0
    assert extractor["transport_residual_tokens"] == 200 - sum(extractor["field_estimated_tokens"].values())
    assert extractor["projected_output_tokens"] == len(json.dumps(project_extractor_payload(json.loads(attempt["raw_output"])), ensure_ascii=False, separators=(",", ":")))
    assert result["facts"] == 1
    assert extractor["output_classes"]["successful_content"]["calls"] == 1


def test_coverage_recovery_is_accounted_separately():
    call, attempt = extractor_call()
    call["purpose"] = "coverage_recovery"
    stories = [{"case_id": "SL-001", "result": {"benchmark_stages": {"extraction": {"events": [], "calls": [{"attempts": [attempt]}]}, "judge": {"batch_reports": []}}}}]
    result = analyze_generation(stories, [call], count_chars)
    assert result["components"]["extractor"]["output_classes"]["coverage_recovery"]["actual_output_tokens"] == 200


def test_schema_failed_but_parseable_extractor_output_is_failed_cost():
    call, attempt = extractor_call()
    call["attempt_status"] = "error"
    stories = [{"case_id": "SL-001", "result": {"benchmark_stages": {"extraction": {"events": [], "calls": [{"attempts": [attempt]}]}, "judge": {"batch_reports": []}}}}]
    result = analyze_generation(stories, [call], count_chars)
    assert result["components"]["extractor"]["output_classes"]["failed_output"]["actual_output_tokens"] == 200


def test_unparseable_output_is_conservatively_unprojectable():
    call, attempt = extractor_call(raw="{bad", output_tokens=17, status="error")
    stories = [{"case_id": "SL-001", "result": {"benchmark_stages": {"extraction": {"events": [], "calls": [{"attempts": [attempt]}]}, "judge": {"batch_reports": []}}}}]
    result = analyze_generation(stories, [call], count_chars)
    extractor = result["components"]["extractor"]
    assert extractor["unprojectable_output_tokens"] == 17
    assert extractor["projected_output_tokens"] == 17
    assert extractor["parse_failures"] == 1


def test_judge_fields_are_split_by_verdict_and_projection_is_estimate():
    raw = judge_raw("contradiction")
    call = {"call_id": "LLM-J", "component": "judge", "output_tokens": 300, "decode_time": 12.0, "total_time": 13.0, "input_tokens": 100}
    stories = [{"case_id": "SL-001", "result": {"benchmark_stages": {
        "extraction": {"events": [], "calls": []},
        "judge": {"batch_reports": [{
            "rows": [{"request_id": "E1", "attempt_records": [{"attempt": 1, "status": "ok", "raw_output": raw}]}],
            "batch_calls": [{"attempt": 1, "request_ids": ["E1"], "timing": {"call_id": "LLM-J"}}],
        }]},
    }}}]
    result = analyze_generation(stories, [call], count_chars)
    judge = result["components"]["judge"]
    assert judge["verdicts"]["contradiction"]["rows"] == 1
    assert judge["field_estimated_tokens"]["reason"] > judge["field_estimated_tokens"]["verdict"]
    assert judge["projection_kind"] == "counterfactual_estimate"
    assert judge["projected_output_tokens"] < judge["actual_output_tokens"]

