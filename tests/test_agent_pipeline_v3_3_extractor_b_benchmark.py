import json

from agent_pipeline_v3_3.extractor_b_benchmark import summarize_attempts
from agent_pipeline_v3_3.extractor_b_cli import build_parser, record_model_load


def test_attempt_summary_separates_accepted_partial_from_failed_generation():
    rows = [{"result": {"benchmark_stages": {"extraction": {
        "calls": [{"attempts": [
            {"status": "ok", "coverage_pending": ["S2"], "timing": {"call_id": "LLM-1"}},
            {"status": "ok", "purpose": "recover_missing_targets", "coverage_pending": [],
             "timing": {"call_id": "LLM-2"}},
        ]}], "uncovered_span_ids": [],
    }}}}]
    calls = [
        {"call_id": "LLM-1", "input_tokens": 100, "output_tokens": 30, "total_time": 3.0},
        {"call_id": "LLM-2", "input_tokens": 20, "output_tokens": 5, "total_time": 0.5},
    ]
    summary = summarize_attempts(rows, calls)
    assert summary["accepted_partial_calls"] == 1
    assert summary["accepted_partial_seconds"] == 3.0
    assert summary["failed_generation_calls"] == 0
    assert summary["local_recovery_calls"] == 1
    assert summary["uncovered_span_count"] == 0


def test_b_cli_exposes_run_and_offline_summary():
    parser = build_parser()
    assert parser.parse_args(["run", "--device", "cuda"]).command == "run"
    assert parser.parse_args(["summary", "--result-dir", "result"]).command == "summary"


def test_model_load_is_persisted_before_story_rows(tmp_path):
    path = tmp_path / "run_metadata.json"
    record_model_load(path, 3.5)
    record_model_load(path, 2.0)
    assert json.loads(path.read_text(encoding="utf-8")) == {"model_load_seconds": 5.5}
