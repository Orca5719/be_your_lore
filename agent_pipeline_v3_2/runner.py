from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import time

from agent_pipeline_v3.metrics import enrich_story_calls
from .lean_judge import judge_frozen_retrieval


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _summary(judge: dict) -> dict:
    counts = {key: 0 for key in ("consistent", "contradiction", "uncertain")}
    for item in judge["items"]:
        counts[item["verdict"]] += 1
    verdict = "contradiction" if counts["contradiction"] else "uncertain" if counts["uncertain"] else "consistent"
    return {"verdict": verdict, "counts": counts, "checked_event_count": len(judge["items"])}


def build_lean_row(source_row: dict, judge: dict, calls: list[dict], elapsed: float) -> dict:
    source_result = source_row["result"]
    result = copy.deepcopy(source_result)
    result["judge"] = judge
    result["benchmark_stages"]["judge"] = judge
    result["summary"] = _summary(judge)
    result["findings"] = [
        {"id": f"R{number}", "event_id": item["event_id"], "event_ids": [item["event_id"]],
         "actors": item["event"].get("actors", []), "event": item["event"].get("event"),
         "verdict": item["verdict"], "label": item["label"], "citations": item["citations"],
         "status": item["status"], "origin": item["origin"]}
        for number, item in enumerate(judge["items"], 1)
    ]
    row = {
        "case_id": source_row["case_id"], "system": "benchmark-3.2-lean-judge",
        "status": judge["status"], "result": result,
        "stage_metrics": {"extraction": {"seconds": 0.0}, "retrieval": {"seconds": 0.0},
                          "judge": {"seconds": elapsed}, "report": {"seconds": 0.0}, "total": {"seconds": elapsed}},
    }
    row["calls"] = enrich_story_calls(row, calls)
    return row


def run_rows(source_rows: list[dict], llm, output: Path, batch_size: int, progress=None) -> list[dict]:
    path = output / "judge_runs.jsonl"
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {row["case_id"] for row in existing}
    prior_calls = [call for row in existing for call in row.get("calls", [])]
    llm.collector.resume_after(prior_calls)
    for source_row in source_rows:
        if source_row["case_id"] in completed:
            continue
        llm.set_story_id(source_row["case_id"])
        before = len(llm.collector.records)
        started = time.perf_counter()
        judge = judge_frozen_retrieval(source_row["result"]["benchmark_stages"]["retrieval"], llm, batch_size, progress)
        elapsed = time.perf_counter() - started
        calls = llm.collector.records[before:]
        existing.append(build_lean_row(source_row, judge, calls, elapsed))
        write_jsonl_atomic(path, existing)
        if progress:
            progress({"purpose": "story_complete", "case_id": source_row["case_id"], "completed": len(existing), "total": len(source_rows)})
    return existing
