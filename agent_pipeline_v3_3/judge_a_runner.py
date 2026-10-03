"""Resumable frozen-input runner for Benchmark 3.3A."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from agent_pipeline_v3.metrics import enrich_story_calls
from agent_pipeline_v3.model import MODEL, REVISION
from agent_pipeline_v3_2.runner import write_jsonl_atomic

from .judge_a import VARIANTS, judge_frozen_retrieval


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(source_hashes: dict[str, str], device: str) -> dict:
    if device != "cuda":
        raise ValueError("Benchmark 3.3A正式运行固定CUDA")
    return {
        "schema_version": "agent-pipeline-v3.3a-manifest-v1",
        "source_hashes": dict(source_hashes),
        "model": MODEL, "revision": REVISION, "device": device, "batch_size": 8,
        "prompt_hashes": {variant: _sha(ROOT / "agent_pipeline_v3_3" / "prompts" / f"judge_a_{variant}.txt")
                          for variant in VARIANTS},
        "implementation_hashes": {name: _sha(ROOT / "agent_pipeline_v3_3" / name)
                                  for name in ("judge_a.py", "judge_a_runner.py", "judge_a_report.py")},
    }


def build_candidate_row(source_row: dict, judge: dict, calls: list[dict], elapsed: float, variant: str) -> dict:
    result = copy.deepcopy(source_row["result"])
    result["judge"] = judge
    result["benchmark_stages"]["judge"] = judge
    counts = {label: sum(item["verdict"] == label for item in judge["items"])
              for label in ("consistent", "contradiction", "uncertain")}
    verdict = "contradiction" if counts["contradiction"] else "uncertain" if counts["uncertain"] else "consistent"
    result["summary"] = {"verdict": verdict, "counts": counts, "checked_event_count": len(judge["items"])}
    result["findings"] = [
        {"id": f"R{number}", "event_id": item["event_id"], "event_ids": [item["event_id"]],
         "actors": item["event"].get("actors", []), "event": item["event"].get("event"),
         "verdict": item["verdict"], "label": item["label"], "citations": item["citations"],
         "status": item["status"], "origin": item["origin"], **({"reason": item["reason"]} if "reason" in item else {})}
        for number, item in enumerate(judge["items"], 1)
    ]
    row = {"case_id": source_row["case_id"], "system": f"benchmark-3.3a-{variant}",
           "status": judge["status"], "result": result,
           "stage_metrics": {"extraction": {"seconds": 0.0}, "retrieval": {"seconds": 0.0},
                             "judge": {"seconds": elapsed}, "report": {"seconds": 0.0},
                             "total": {"seconds": elapsed}}}
    row["calls"] = enrich_story_calls(row, calls)
    return row


def run_variant(source_rows: list[dict], llm, directory: Path, variant: str, progress=None) -> list[dict]:
    if variant not in VARIANTS:
        raise ValueError("未知Judge实验版本")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "judge_runs.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    ids = [row["case_id"] for row in existing]
    source_ids = [row["case_id"] for row in source_rows]
    if ids != source_ids[:len(ids)]:
        raise ValueError("续跑案例顺序或ID与冻结输入不符")
    llm.collector.resume_after([call for row in existing for call in row.get("calls", [])])
    for source_row in source_rows[len(existing):]:
        llm.set_story_id(source_row["case_id"])
        before = len(llm.collector.records)
        started = time.perf_counter()
        judge = judge_frozen_retrieval(source_row["result"]["benchmark_stages"]["retrieval"], llm, variant, 8)
        elapsed = time.perf_counter() - started
        calls = llm.collector.records[before:]
        existing.append(build_candidate_row(source_row, judge, calls, elapsed, variant))
        write_jsonl_atomic(path, existing)
        if progress:
            progress(variant, source_row["case_id"], len(existing), len(source_rows))
    return existing
