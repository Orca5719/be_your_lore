import json


def extraction():
    return {
        "schema_version": "agent-pipeline-v2-extraction-v1",
        "prompt_version": "extractor-v1",
        "stage": "extraction",
        "status": "ok",
        "text": "雷的右侧心脏寄宿亚巴顿。",
        "spans": {"S1": {"text": "雷的右侧心脏寄宿亚巴顿。", "start": 0, "end": 13}},
        "events": [{"id": "E1", "actors": ["雷"], "event": "雷的右侧心脏寄宿亚巴顿", "mental_state": None, "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism"}],
        "ignored_spans": [],
        "non_event_span_ids": [],
        "uncovered_span_ids": [],
        "rejected": [],
        "calls": [],
        "processing_complete": True,
        "model_loaded_this_request": False,
        "device": "cpu",
        "model_load_seconds": 0,
        "request_seconds": 0,
    }


def test_event_fact_preserves_subject_and_maps_check_dimensions():
    from agent_pipeline_v2_1.story_pipeline import event_fact

    value = event_fact(extraction()["events"][0])
    assert value["subject"] == "雷"
    assert value["normalized_fact"] == "雷的右侧心脏寄宿亚巴顿"
    assert value["dimension"] == "physical_rule"
    assert value["dimensions"] == ["physical_rule", "world_rule"]


def test_story_retrieval_uses_one_structured_query_per_event():
    from agent_pipeline_v2_1.story_pipeline import retrieve_story_events

    class FakeRetriever:
        def search_fact(self, fact, query, **kwargs):
            assert fact["subject"] == "雷"
            return {"method": kwargs["method"], "metadata_filter_enabled": kwargs["metadata_filter"], "filter": None, "results": [{"id": "l1", "text": "亚巴顿寄宿在雷的左侧心脏里。", "file": "a", "start_line": 1, "end_line": 1, "heading_path": []}]}

    report = retrieve_story_events(extraction(), FakeRetriever(), method="hybrid", metadata_filter=False, k=5)
    assert report["status"] == "ok"
    assert report["items"][0]["event_id"] == "E1"
    assert report["items"][0]["evidence"][0]["id"] == "l1"


def test_story_judge_applies_v2_1_coexistence_guard_and_builds_report():
    from agent_pipeline_v2_1.story_pipeline import judge_story_events, build_story_report

    retrieval = {
        "status": "ok",
        "extraction": extraction(),
        "events": extraction()["events"],
        "items": [{"event_id": "E1", "event": extraction()["events"][0], "fact": {"subject": "雷", "normalized_fact": "雷的右侧心脏寄宿亚巴顿", "dimension": "physical_rule", "dimensions": ["physical_rule"]}, "status": "ok", "evidence": [{"id": "l1", "text": "亚巴顿寄宿在雷的左侧心脏里。", "file": "a", "start_line": 1, "end_line": 1, "heading_path": []}]}],
    }

    class FakeLLM:
        last_batch_generation = {}

        def _generate_batch(self, messages, max_new_tokens=768):
            self.last_batch_generation = {"seconds": 0.01, "generated_tokens": [10] * len(messages)}
            value = {"verdict": "contradiction", "citations": [{"evidence_id": "L1", "quote": "亚巴顿寄宿在雷的左侧心脏里。"}], "reason": "左右位置冲突。", "assessment": {"same_subject": True, "evidence_applicable": True, "relation": "direct_conflict", "can_both_be_true": False, "assumptions": []}}
            return [json.dumps(value, ensure_ascii=False) for _ in messages]

    judged = judge_story_events(retrieval, FakeLLM(), batch_size=8)
    report = build_story_report(judged)
    assert judged["items"][0]["verdict"] == "contradiction"
    assert judged["items"][0]["citations"][0]["chunk_id"] == "l1"
    assert report["summary"]["verdict"] == "contradiction"
    assert report["findings"][0]["event_ids"] == ["E1"]


def test_story_cli_defaults_to_hybrid_without_metadata_filter():
    from agent_pipeline_v2_1.cli import build_parser

    args = build_parser().parse_args(["run-stories", "--device", "cuda"])
    assert args.method == "hybrid"
    assert args.metadata_filter is False
    assert args.judge_batch_size == 8
    assert args.top_k == 5
    assert args.reuse_ok_from is None
