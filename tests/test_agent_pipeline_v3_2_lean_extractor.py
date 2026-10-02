from __future__ import annotations

import json

import pytest

from agent_pipeline_v3_2.lean_extractor import _sanitize_recovery_value, decode_lean_window, extract_events_lean


def spans():
    return {
        "S1": {"text": "雷捂住左胸，", "start": 0, "end": 7},
        "S2": {"text": "感到亚巴顿开始躁动。", "start": 7, "end": 18},
    }


def test_lean_wire_restores_canonical_defaults_and_ignored_reason():
    value = {
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed",
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_span_ids": ["S2"],
        "non_event_span_ids": [],
    }
    events, ignored, non_events = decode_lean_window(value, spans(), {"target_ids": ["S1", "S2"], "context_ids": []}, 1)
    assert events[0]["explicit"] is True
    assert events[0]["mental_state"] is None
    assert events[0]["conditions"] == []
    assert ignored == [{"source_id": "S2", "reason": "process_detail"}]
    assert non_events == []


def test_inferred_derives_explicit_false_and_optional_fields_survive():
    value = {
        "events": [{
            "actors": ["雷"], "event": "雷可能感到异常", "modality": "inferred",
            "mental_state": "不安", "conditions": ["左胸疼痛"],
            "source_ids": ["S1"], "context_ids": [], "check_reason": "consequence_support",
        }],
        "ignored_span_ids": [], "non_event_span_ids": ["S2"],
    }
    events, _, _ = decode_lean_window(value, spans(), {"target_ids": ["S1", "S2"], "context_ids": []}, 1)
    assert events[0]["explicit"] is False
    assert events[0]["mental_state"] == "不安"
    assert events[0]["conditions"] == ["左胸疼痛"]


def test_lean_wire_rejects_uncovered_target():
    value = {"events": [], "ignored_span_ids": ["S1"], "non_event_span_ids": []}
    with pytest.raises(ValueError, match="未覆盖"):
        decode_lean_window(value, spans(), {"target_ids": ["S1", "S2"], "context_ids": []}, 1)


def test_lean_wire_rejects_removed_or_extra_fields():
    value = {
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed", "explicit": True,
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_span_ids": ["S2"], "non_event_span_ids": [],
    }
    with pytest.raises(ValueError, match="Lean wire"):
        decode_lean_window(value, spans(), {"target_ids": ["S1", "S2"], "context_ids": []}, 1)


class FakeLLM:
    device = "cuda"
    load_seconds = 0.0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens=1536):
        self.last_generation = {"call_id": f"LLM-{len(self.outputs):06d}", "generated_tokens": 10, "seconds": 0.1}
        return self.outputs.pop(0)


def test_extract_events_lean_returns_existing_canonical_report_schema():
    raw = json.dumps({
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed",
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_span_ids": [], "non_event_span_ids": [],
    }, ensure_ascii=False)
    result = extract_events_lean("雷捂住左胸，", device="cuda", llm=FakeLLM([raw]))
    assert result["schema_version"] == "agent-pipeline-v2-extraction-v1"
    assert result["prompt_version"] == "extractor-lean-v2"
    assert result["status"] == "ok"
    assert result["events"][0]["explicit"] is True


def test_missing_coverage_uses_lean_recovery_payload():
    first = json.dumps({
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed",
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_span_ids": [], "non_event_span_ids": [],
    }, ensure_ascii=False)
    recovery = json.dumps({
        "events": [], "ignored_span_ids": ["S2"], "non_event_span_ids": [],
    }, ensure_ascii=False)
    result = extract_events_lean("雷捂住左胸，喝了一口水。", device="cuda", llm=FakeLLM([first, recovery]))
    assert result["status"] == "ok"
    assert result["uncovered_span_ids"] == []
    assert result["calls"][0]["recovered_target_ids"] == ["S2"]
    assert len(result["calls"][0]["attempts"]) == 2



def test_recovery_drops_dispositions_outside_recovery_targets():
    value = {
        "events": [],
        "ignored_span_ids": ["S8"],
        "non_event_span_ids": ["S3"],
    }
    sanitized, changes = _sanitize_recovery_value(value, {"target_ids": ["S1", "S8"], "context_ids": ["S6", "S7"]})
    assert sanitized["ignored_span_ids"] == ["S8"]
    assert sanitized["non_event_span_ids"] == []
    assert changes == ["recovery_out_of_scope_disposition_removed"]


def test_partial_recovery_retries_only_the_remaining_target():
    class RecordingLLM(FakeLLM):
        def __init__(self, outputs):
            super().__init__(outputs)
            self.payloads = []

        def _generate(self, messages, max_new_tokens=1536):
            self.payloads.append(json.loads(messages[-1]["content"]))
            return super()._generate(messages, max_new_tokens=max_new_tokens)

    first = json.dumps({"events": [], "ignored_span_ids": [], "non_event_span_ids": []})
    partial = json.dumps({"events": [], "ignored_span_ids": ["S2"], "non_event_span_ids": []})
    final = json.dumps({"events": [], "ignored_span_ids": ["S1"], "non_event_span_ids": []})
    llm = RecordingLLM([first, partial, final])
    result = extract_events_lean("雷捂住左胸，喝了一口水。", device="cuda", llm=llm)
    assert result["status"] == "ok"
    assert result["uncovered_span_ids"] == []
    assert {row["source_id"] for row in result["ignored_spans"]} == {"S1", "S2"}
    assert list(llm.payloads[2]["target_spans"]) == ["S1"]
    assert len(result["calls"][0]["attempts"]) == 3


def test_failed_followup_keeps_valid_events_and_reports_only_missing_span():
    first = json.dumps({
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed",
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }],
        "ignored_span_ids": [], "non_event_span_ids": [],
    }, ensure_ascii=False)
    no_coverage = json.dumps({"events": [], "ignored_span_ids": [], "non_event_span_ids": []})
    result = extract_events_lean(
        "雷捂住左胸，喝了一口水。", device="cuda",
        llm=FakeLLM([first, no_coverage, no_coverage, '{"events":[],"ignored_span_ids":[],"non_event_span_ids":[],"extra":[]}', '{"disposition":"event"}']),
    )
    assert result["status"] == "partial"
    assert result["uncovered_span_ids"] == ["S2"]
    assert result["events"][0]["event"] == "雷捂住左胸"


def test_single_span_disposition_recovers_ordinary_scene_without_losing_event():
    first = json.dumps({
        "events": [{
            "actors": ["雷"], "event": "雷捂住左胸", "modality": "observed",
            "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism",
        }], "ignored_span_ids": [], "non_event_span_ids": [],
    }, ensure_ascii=False)
    no_coverage = '{"events":[],"ignored_span_ids":[],"non_event_span_ids":[]}'
    invalid_retry = '{"events":[],"ignored_span_ids":[],"non_event_span_ids":[],"extra":[]}'
    llm = FakeLLM([first, no_coverage, no_coverage, invalid_retry, '{"disposition":"ignored"}'])
    result = extract_events_lean("雷捂住左胸，喝了一口水。", device="cuda", llm=llm)
    assert result["status"] == "ok"
    assert result["uncovered_span_ids"] == []
    assert result["events"][0]["event"] == "雷捂住左胸"
    assert result["ignored_spans"] == [{"source_id": "S2", "reason": "process_detail"}]
    assert result["calls"][0]["attempts"][-1]["purpose"] == "recover_disposition"
