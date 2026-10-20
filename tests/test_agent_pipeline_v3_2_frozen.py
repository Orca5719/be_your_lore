from __future__ import annotations

import json

import pytest

from agent_pipeline_v3_2.frozen import load_and_validate_source


def test_source_validation_requires_24_cases_and_80_facts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "story_runs.jsonl").write_text(
        json.dumps({"case_id": "SL-001", "result": {"benchmark_stages": {"retrieval": {"items": []}}}}) + "\n",
        encoding="utf-8",
    )
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    (source / "profile_summary.json").write_text("{}", encoding="utf-8")
    (source / "quality_reference.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="24"):
        load_and_validate_source(source)
