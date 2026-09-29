import json

import pytest

from agent_pipeline_v3.profiling import TraceCollector
from agent_pipeline_v3.runner import build_manifest, ensure_manifest, load_story_rows, run_profile_stories


class FakeLLM:
    def __init__(self):
        self.collector = TraceCollector()
        self.load_seconds = 7.5

    def set_story_id(self, story_id):
        self.collector.set_story(story_id)

    def clear_traces(self):
        self.collector.clear()


def fake_process(llm, seen):
    def process(system, case_id, text, progress=None):
        seen.append(case_id)
        span = llm.collector.start("extractor", 1, 10, 0)
        span.mark_first_token()
        trace = span.finish(2)
        return {
            "case_id": case_id,
            "system": "candidate",
            "status": "ok",
            "stage_metrics": {
                "extraction": {"seconds": trace["total_time"]},
                "retrieval": {"seconds": 0},
                "judge": {"seconds": 0},
                "report": {"seconds": 0},
                "total": {"seconds": trace["total_time"]},
            },
            "result": {
                "judge": {"items": []},
                "benchmark_stages": {
                    "extraction": {
                        "calls": [
                            {
                                "window_id": 1,
                                "purpose": "extract_checkable_events",
                                "target_ids": ["S1"],
                                "attempts": [{"attempt": 1, "timing": {"call_id": trace["call_id"]}}],
                            }
                        ]
                    },
                    "judge": {"batch_reports": []},
                },
            },
        }

    return process


def test_runner_excludes_warmup_persists_calls_and_resumes(tmp_path):
    cases = [{"id": "SL-001", "story": "a"}, {"id": "SL-002", "story": "b"}]
    llm = FakeLLM()
    seen = []
    rows = run_profile_stories(cases, llm, object(), tmp_path, process=fake_process(llm, seen))

    assert seen == ["SL-001", "SL-001", "SL-002"]
    assert [row["case_id"] for row in rows] == ["SL-001", "SL-002"]
    assert all(row["calls"][0]["story_id"] == row["case_id"] for row in rows)
    trace_rows = [json.loads(line) for line in (tmp_path / "inference_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(trace_rows) == 2
    assert all(row["story_id"] != "WARMUP" for row in trace_rows)

    resumed_seen = []
    resumed_llm = FakeLLM()
    resumed = run_profile_stories(cases, resumed_llm, object(), tmp_path, process=fake_process(resumed_llm, resumed_seen))
    assert resumed_seen == []
    assert resumed == rows


def test_load_story_rows_rejects_duplicate_cases(tmp_path):
    path = tmp_path / "story_runs.jsonl"
    row = {"case_id": "SL-001"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_story_rows(path)


def test_manifest_rejects_changed_identity(tmp_path):
    path = tmp_path / "manifest.json"
    first = build_manifest(dataset={"cases": []}, inputs={"a": "1"}, device="cuda")
    ensure_manifest(path, first)
    changed = build_manifest(dataset={"cases": []}, inputs={"a": "2"}, device="cuda")
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        ensure_manifest(path, changed)
