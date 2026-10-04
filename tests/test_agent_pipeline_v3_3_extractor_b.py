import json

from agent_pipeline_v2.extractor import extract_events as baseline_extract
from agent_pipeline_v3_3.extractor_b import extract_events_b


def event(source_id="S1"):
    return {
        "actors": ["雷"], "event": "雷拥有两颗心脏", "mental_state": None,
        "explicit": True, "modality": "observed", "conditions": [],
        "source_ids": [source_id], "context_ids": [], "check_reason": "mechanism",
    }


class FakeLLM:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self, answers):
        self.answers = iter(answers)
        self.messages = []
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens):
        self.messages.append(messages)
        self.last_generation = {"seconds": 0.1, "input_tokens": 10, "generated_tokens": 5}
        answer = next(self.answers)
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def wire(*, events=None, ignored=None, non_events=None):
    return {"events": events or [], "ignored_spans": ignored or [], "non_event_span_ids": non_events or []}


def test_missing_coverage_keeps_first_output_and_recovers_only_missing_span():
    llm = FakeLLM([wire(events=[event()]), wire(ignored=[{"source_id": "S2", "reason": "routine"}])])
    result = extract_events_b("雷拥有两颗心脏。他喝水。", llm=llm)
    assert result["status"] == "ok"
    assert [row["source_id"] for row in result["ignored_spans"]] == ["S2"]
    assert result["events"][0]["event"] == "雷拥有两颗心脏"
    assert result["coverage_accounting"]["complete"]
    assert result["calls"][0]["attempts"][0]["status"] == "ok"
    assert result["calls"][0]["attempts"][0]["coverage_pending"] == ["S2"]
    assert result["calls"][0]["attempts"][1]["purpose"] == "recover_missing_targets"
    assert list(json.loads(llm.messages[1][1]["content"])["target_spans"]) == ["S2"]


def test_failed_local_recovery_retains_valid_first_event_and_marks_uncovered():
    llm = FakeLLM([wire(events=[event()]), "not json", "still not json"])
    result = extract_events_b("雷拥有两颗心脏。他喝水。", llm=llm)
    assert result["status"] == "partial"
    assert len(result["events"]) == 1
    assert result["uncovered_span_ids"] == ["S2"]
    assert len(llm.messages) == 3
    assert all(list(json.loads(messages[1]["content"])["target_spans"]) == ["S2"] for messages in llm.messages[1:])


def test_invalid_schema_retries_full_window_once():
    llm = FakeLLM([{"events": []}, wire(events=[event()], non_events=["S2"])])
    result = extract_events_b("雷拥有两颗心脏。随后。", llm=llm)
    assert result["status"] == "ok"
    assert len(llm.messages) == 2
    assert result["calls"][0]["attempts"][0]["status"] == "error"


def test_complete_first_response_needs_no_recovery():
    answer = wire(events=[event()], non_events=["S2"])
    llm = FakeLLM([answer])
    result = extract_events_b("雷拥有两颗心脏。随后。", llm=llm)
    baseline = baseline_extract("雷拥有两颗心脏。随后。", llm=FakeLLM([answer]))
    assert result["status"] == "ok"
    assert len(llm.messages) == 1
    assert result["coverage_accounting"]["uncovered_span_ids"] == []
    for field in ("events", "ignored_spans", "non_event_span_ids", "uncovered_span_ids"):
        assert result[field] == baseline[field]


def test_uncovered_ids_match_canonical_order_across_windows():
    story = "。".join("雷" + str(i) for i in range(1, 13)) + "。"
    answers = [
        wire(events=[{**event(), "event": "雷1有异常", "actors": ["雷1"]}],
             non_events=[f"S{i}" for i in range(3, 9)]),
        "invalid", "invalid",
        wire(non_events=["S9", "S11", "S12"]),
        "invalid", "invalid",
    ]
    result = extract_events_b(story, llm=FakeLLM(answers))
    assert result["status"] == "partial"
    assert result["uncovered_span_ids"] == ["S10", "S2"]
    assert set(result["coverage_accounting"]["uncovered_span_ids"]) == {"S2", "S10"}
