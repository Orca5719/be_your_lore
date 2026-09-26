import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_pipeline_v3.runner import build_manifest, ensure_manifest, load_jsonl, run_resumable


def test_manifest_rejects_changed_identity(tmp_path):
    first = build_manifest(dataset={"a": 1}, configs=[], inputs={"p": "1"}, repeats=3, warmup=1)
    ensure_manifest(tmp_path / "manifest.json", first)
    second = build_manifest(dataset={"a": 2}, configs=[], inputs={"p": "1"}, repeats=3, warmup=1)
    with pytest.raises(ValueError, match="hash mismatch"):
        ensure_manifest(tmp_path / "manifest.json", second)


def test_resume_skips_completed_and_rejects_duplicates(tmp_path):
    config = SimpleNamespace(name="baseline")
    calls = []
    def process(system, case_id, text):
        calls.append(case_id)
        metrics = {name: {"seconds": 0.0} for name in ("extraction", "retrieval", "judge", "report", "total")}
        return {"case_id": case_id, "system": "baseline", "status": "ok", "result": {}, "stage_metrics": metrics}
    path = tmp_path / "rows.jsonl"
    rows = run_resumable([{"id": "C1", "story": "x"}], [SimpleNamespace(config=config)], path, process)
    assert len(rows) == 1 and calls == ["C1"]
    run_resumable([{"id": "C1", "story": "x"}], [SimpleNamespace(config=config)], path, process)
    assert calls == ["C1"]
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)
