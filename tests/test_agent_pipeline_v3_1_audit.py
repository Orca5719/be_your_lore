import json
from pathlib import Path

import pytest

from agent_pipeline_v3_1.audit import audit_retries, validate_result


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def call(call_id, component, *, retry=False, output=10, seconds=1.0):
    return {
        "call_id": call_id,
        "story_id": "SL-001",
        "component": component,
        "status": "ok",
        "input_tokens": 100,
        "padded_input_tokens": 0,
        "compute_input_tokens": 100,
        "output_tokens": output,
        "prefill_time": 0.1,
        "decode_time": seconds - 0.1,
        "total_time": seconds,
        "ttft_ms": 100.0,
        "batch_size": 1,
        "peak_allocated": 1,
        "peak_reserved": 2,
        "is_retry": retry,
    }


def result_fixture(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    calls = [
        call("LLM-000001", "extractor"),
        call("LLM-000002", "extractor", retry=True, output=20, seconds=2.0),
        call("LLM-000003", "judge"),
        call("LLM-000004", "judge", retry=True, output=30, seconds=3.0),
    ]
    row = {
        "case_id": "SL-001",
        "status": "ok",
        "calls": calls,
        "result": {
            "benchmark_stages": {
                "extraction": {
                    "calls": [{
                        "window_id": 1,
                        "purpose": "extract_checkable_events",
                        "attempts": [
                            {"attempt": 1, "status": "error", "error_type": "ValueError", "error": "存在未覆盖的target_spans：S1", "raw_output": "{}", "timing": {"call_id": "LLM-000001"}},
                            {"attempt": 2, "purpose": "recover_missing_targets", "status": "ok", "raw_output": "{}", "timing": {"call_id": "LLM-000002"}},
                        ],
                    }],
                },
                "judge": {
                    "batch_reports": [{
                        "rows": [{"request_id": "E1", "attempt_records": [
                            {"attempt": 1, "status": "error", "error_type": "ValueError", "error": "citation字段无效", "raw_output": "{}"},
                            {"attempt": 2, "status": "ok", "raw_output": "{}"},
                        ]}],
                        "batch_calls": [
                            {"attempt": 1, "request_ids": ["E1"], "timing": {"call_id": "LLM-000003"}},
                            {"attempt": 2, "request_ids": ["E1"], "timing": {"call_id": "LLM-000004"}},
                        ],
                    }],
                },
            },
        },
    }
    write_jsonl(source / "story_runs.jsonl", [row])
    write_jsonl(source / "inference_calls.jsonl", calls)
    (source / "profile_summary.json").write_text(json.dumps({"quality_guard": {"comparable": True, "matched_cases": 1, "total_reference_cases": 1}}), encoding="utf-8")
    return source


def test_validate_reconciles_story_and_flat_call_records(tmp_path):
    source = result_fixture(tmp_path)
    summary = validate_result(source, expected_stories=1, expected_calls=4, expected_extractor=2, expected_judge=2)
    assert summary["status"] == "ok"
    assert summary["quality_matched"] == 1


def test_validate_rejects_orphan_flat_call(tmp_path):
    source = result_fixture(tmp_path)
    rows = [json.loads(line) for line in (source / "inference_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[-1]["call_id"] = "LLM-999999"
    write_jsonl(source / "inference_calls.jsonl", rows)
    with pytest.raises(ValueError, match="call records do not reconcile"):
        validate_result(source, expected_stories=1, expected_calls=4, expected_extractor=2, expected_judge=2)


def test_validate_rejects_attempt_call_mapping_gap(tmp_path):
    source = result_fixture(tmp_path)
    rows = [json.loads(line) for line in (source / "story_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["result"]["benchmark_stages"]["extraction"]["calls"][0]["attempts"][0]["timing"]["call_id"] = "LLM-999999"
    write_jsonl(source / "story_runs.jsonl", rows)
    nested_calls = rows[0]["calls"]
    write_jsonl(source / "inference_calls.jsonl", nested_calls)
    with pytest.raises(ValueError, match="attempt/call mapping incomplete"):
        validate_result(source, expected_stories=1, expected_calls=4, expected_extractor=2, expected_judge=2)


def test_audit_separates_recovery_and_judge_retry_and_preserves_source(tmp_path):
    source = result_fixture(tmp_path)
    output = tmp_path / "audit"
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    report = audit_retries(source, output, expected_stories=1, expected_calls=4, expected_extractor=2, expected_judge=2)

    assert report["retry_calls"] == 2
    assert report["categories"]["MISSING_TARGET_COVERAGE"]["calls"] == 1
    assert report["categories"]["BATCH_ROW_RETRY"]["calls"] == 1
    assert report["retry_kinds"]["coverage_recovery"]["calls"] == 1
    assert report["retry_kinds"]["judge_row_retry"]["calls"] == 1
    assert report["judge_retry_causes"]["INVALID_FIELD_OR_REFERENCE"]["calls"] == 1
    assert report["totals"]["output_tokens"] == 50
    assert report["totals"]["total_time"] == pytest.approx(5.0)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert (output / "calls_audit.jsonl").exists()
    assert (output / "retry_taxonomy.json").exists()
    assert (output / "retry_taxonomy.csv").exists()


def test_audit_rejects_output_inside_frozen_source(tmp_path):
    source = result_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen source"):
        audit_retries(source, source / "audit", expected_stories=1, expected_calls=4, expected_extractor=2, expected_judge=2)
