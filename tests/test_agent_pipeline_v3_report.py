import csv
import json

from agent_pipeline_v3.report import render_markdown, write_profile_outputs


def profile():
    component = {
        "calls": 1,
        "retry_calls": 0,
        "avg_batch_size": 1.0,
        "input_tokens": 10,
        "padded_input_tokens": 0,
        "compute_input_tokens": 10,
        "output_tokens": 2,
        "prefill_seconds": 1.0,
        "decode_seconds": 2.0,
        "total_seconds": 3.0,
        "ttft_p50_ms": 1000.0,
        "ttft_p95_ms": 1000.0,
        "ttft_max_ms": 1000.0,
        "prefill_useful_tokens_per_second": 10.0,
        "prefill_compute_tokens_per_second": 10.0,
        "decode_tokens_per_second": 1.0,
        "peak_allocated": 1024,
        "peak_reserved": 2048,
        "llm_time_share": 1.0,
        "status_counts": {"ok": 1, "error": 0},
    }
    return {
        "schema_version": "agent-pipeline-v3-profile-summary-v1",
        "components": {"extractor": component, "judge": {**component, "calls": 0}},
        "workflow": {
            "model_load_seconds": 5.0,
            "stage_seconds": {"extraction": 3.2, "retrieval": 0.2, "judge": 0.0, "report": 0.1, "total": 4.0},
            "llm_generation_seconds": 3.0,
            "non_llm_seconds": 1.0,
            "end_to_end_seconds": 4.0,
            "story_count": 1,
            "fact_count": 1,
            "stories_per_second": 0.25,
            "facts_per_second": 0.25,
            "calls_per_story": 1.0,
            "tokens_per_story": 12.0,
            "report_llm_calls": 0,
        },
        "retry_cost": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_time": 0.0},
        "slowest_calls": [],
        "alignment": {"component_llm_seconds": 3.0, "call_llm_seconds": 3.0, "explained_end_to_end_seconds": 4.0},
    }


def test_markdown_contains_component_workflow_and_quality_guard():
    text = render_markdown(profile(), {"comparable": True, "matched_cases": 24, "total_reference_cases": 24})
    assert "Benchmark 3 LLM Workload Profile" in text
    assert "Extractor" in text
    assert "Report" in text
    assert "Quality guard: PASS" in text
    assert "LLM time share" in text
    assert "Timing alignment" in text


def test_write_outputs_creates_json_markdown_and_both_csv_files(tmp_path):
    calls = [{"call_id": "LLM-1", "component": "extractor", "story_id": "SL-1", "total_time": 3.0}]
    write_profile_outputs(tmp_path, profile(), calls, {"comparable": True})
    assert (tmp_path / "profile_summary.json").exists()
    assert (tmp_path / "profile_summary.md").exists()
    assert (tmp_path / "component_metrics.csv").exists()
    assert (tmp_path / "inference_calls.csv").exists()
    data = json.loads((tmp_path / "profile_summary.json").read_text(encoding="utf-8"))
    assert data["quality_guard"]["comparable"] is True
    with (tmp_path / "component_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert {row["component"] for row in csv.DictReader(handle)} == {"extractor", "judge"}
