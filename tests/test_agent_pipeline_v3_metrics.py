import pytest

from agent_pipeline_v3.metrics import aggregate_profile, enrich_story_calls


def call(call_id, component, input_tokens, padding, output, prefill, decode, batch, peak, retry=False):
    return {
        "schema_version": "agent-pipeline-v3-inference-call-v1",
        "call_id": call_id,
        "story_id": "SL-001",
        "component": component,
        "status": "ok",
        "input_tokens": input_tokens,
        "padded_input_tokens": padding,
        "compute_input_tokens": input_tokens + padding,
        "output_tokens": output,
        "prefill_time": prefill,
        "decode_time": decode,
        "total_time": prefill + decode,
        "ttft_ms": prefill * 1000,
        "batch_size": batch,
        "peak_allocated": peak,
        "peak_reserved": peak + 50,
        "error": None,
        "is_retry": retry,
    }


def story_row():
    return {
        "case_id": "SL-001",
        "status": "ok",
        "stage_metrics": {
            "extraction": {"seconds": 4.0},
            "retrieval": {"seconds": 1.0},
            "judge": {"seconds": 11.0},
            "report": {"seconds": 0.5},
            "total": {"seconds": 18.0},
        },
        "result": {"judge": {"items": [{}, {}, {}]}},
    }


def test_aggregate_uses_weighted_throughput_nearest_rank_and_max_vram():
    calls = [
        call("LLM-000001", "extractor", 100, 0, 10, 1.0, 2.0, 1, 100),
        call("LLM-000002", "judge", 300, 100, 30, 2.0, 6.0, 8, 200),
        call("LLM-000003", "judge", 50, 0, 5, 1.0, 1.0, 1, 150, retry=True),
    ]
    profile = aggregate_profile(calls, [story_row()], model_load_seconds=9.0)
    judge = profile["components"]["judge"]

    assert judge["calls"] == 2
    assert judge["retry_calls"] == 1
    assert judge["avg_batch_size"] == pytest.approx(4.5)
    assert judge["decode_tokens_per_second"] == pytest.approx(35 / 7)
    assert judge["prefill_useful_tokens_per_second"] == pytest.approx(350 / 3)
    assert judge["prefill_compute_tokens_per_second"] == pytest.approx(450 / 3)
    assert judge["ttft_p50_ms"] == pytest.approx(1500)
    assert judge["ttft_p95_ms"] == pytest.approx(2000)
    assert judge["peak_allocated"] == 200
    assert profile["workflow"]["llm_generation_seconds"] == pytest.approx(13.0)
    assert profile["workflow"]["non_llm_seconds"] == pytest.approx(5.0)
    assert profile["workflow"]["report_llm_calls"] == 0
    assert profile["workflow"]["facts_per_second"] == pytest.approx(3 / 18)
    assert profile["alignment"]["component_call_delta_seconds"] == pytest.approx(0)
    assert profile["alignment"]["end_to_end_delta_seconds"] == pytest.approx(0)


def test_slowest_calls_and_retry_cost_are_reported():
    calls = [
        call("LLM-000001", "extractor", 1, 0, 1, 1, 1, 1, 10),
        call("LLM-000002", "judge", 1, 0, 1, 2, 5, 1, 20, retry=True),
    ]
    profile = aggregate_profile(calls, [story_row()], model_load_seconds=1.0)
    assert profile["slowest_calls"][0]["call_id"] == "LLM-000002"
    assert profile["retry_cost"] == {
        "calls": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_time": 7,
    }


def test_enrich_story_calls_maps_attempts_and_operations_by_call_id():
    calls = [
        call("LLM-000001", "extractor", 1, 0, 1, 1, 1, 1, 10),
        call("LLM-000002", "judge", 1, 0, 1, 1, 1, 1, 10),
    ]
    row = {
        "result": {
            "benchmark_stages": {
                "extraction": {
                    "calls": [
                        {
                            "window_id": 2,
                            "purpose": "extract_checkable_events",
                            "attempts": [{"attempt": 2, "purpose": "recover_missing_targets", "timing": {"call_id": "LLM-000001"}}],
                        }
                    ]
                },
                "judge": {
                    "batch_reports": [
                        {
                            "batch_calls": [
                                {"attempt": 1, "request_ids": ["E1", "E2"], "timing": {"call_id": "LLM-000002"}}
                            ]
                        }
                    ]
                },
            }
        }
    }
    enriched = enrich_story_calls(row, calls)
    by_id = {item["call_id"]: item for item in enriched}
    assert by_id["LLM-000001"]["is_retry"] is True
    assert by_id["LLM-000001"]["operation"] == "recover_missing_targets"
    assert by_id["LLM-000001"]["window_id"] == 2
    assert by_id["LLM-000002"]["is_retry"] is False
    assert by_id["LLM-000002"]["request_ids"] == ["E1", "E2"]


def test_enrichment_rejects_unmapped_trace():
    with pytest.raises(ValueError, match="not linked"):
        enrich_story_calls({"result": {"benchmark_stages": {"extraction": {"calls": []}, "judge": {"batch_reports": []}}}}, [call("LLM-9", "extractor", 1, 0, 1, 1, 1, 1, 1)])
