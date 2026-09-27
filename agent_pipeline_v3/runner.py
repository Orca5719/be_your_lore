from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from agent_pipeline_v2_2.schema import SystemConfig, stable_digest

from .metrics import enrich_story_calls


MANIFEST_SCHEMA = "agent-pipeline-v3-profile-manifest-v1"


def build_manifest(*, dataset: dict, inputs: dict[str, str], device: str) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    identity = {
        "dataset": stable_digest(dataset),
        "inputs": inputs,
        "device": device,
        "system": SystemConfig.candidate().to_dict(),
        "warmup_stories": 1,
        "repeats": 1,
        "profiling_schema": "agent-pipeline-v3-inference-call-v1",
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": stable_digest(identity),
    }


def ensure_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("identity_sha256") != manifest.get("identity_sha256"):
            raise ValueError("resume configuration hash mismatch")
        return
    atomic_write_json(path, manifest)


def load_story_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("case_id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("duplicate or invalid story rows")
    return rows


def _flatten_calls(rows: list[dict]) -> list[dict]:
    calls = [call for row in rows for call in row.get("calls", [])]
    ids = [call.get("call_id") for call in calls]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate inference call IDs")
    return calls


def _persist(directory: Path, rows: list[dict]) -> None:
    write_jsonl_atomic(directory / "story_runs.jsonl", rows)
    write_jsonl_atomic(directory / "inference_calls.jsonl", _flatten_calls(rows))


def run_profile_stories(cases: list[dict], llm, system, directory: Path, *, process, progress=None) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = load_story_rows(directory / "story_runs.jsonl")
    existing_calls = _flatten_calls(rows)
    llm.collector.resume_after(existing_calls)
    _persist(directory, rows)
    completed = {row["case_id"] for row in rows}
    remaining = [case for case in cases if case["id"] not in completed]
    if not remaining:
        return rows
    warmup = cases[0]
    llm.set_story_id("WARMUP")
    if progress:
        progress({"kind": "warmup", "case_id": warmup["id"]})
    warmup_progress = None if progress is None else lambda step: progress({"kind": "stage", "case_id": "WARMUP", **step})
    process(system, warmup["id"], warmup["story"], progress=warmup_progress)
    llm.clear_traces()
    for case in remaining:
        if progress:
            progress({"kind": "story", "case_id": case["id"], "completed": len(rows), "total": len(cases)})
        llm.set_story_id(case["id"])
        before = len(llm.collector.records)
        stage_progress = None if progress is None else lambda step: progress({"kind": "stage", "case_id": case["id"], **step})
        row = process(system, case["id"], case["story"], progress=stage_progress)
        raw_calls = llm.collector.records[before:]
        stored = {**row, "calls": enrich_story_calls(row, raw_calls)}
        rows.append(stored)
        _persist(directory, rows)
    return rows
