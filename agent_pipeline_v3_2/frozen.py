from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_and_validate_source(source: Path) -> dict:
    source = source.resolve()
    required = [source / "story_runs.jsonl", source / "manifest.json", source / "profile_summary.json", source / "quality_reference.json"]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise ValueError("Benchmark 3 source missing: " + ", ".join(missing))
    rows = read_jsonl(source / "story_runs.jsonl")
    if len(rows) != 24:
        raise ValueError("Benchmark 3.2A requires exactly 24 source stories")
    ids = [row.get("case_id") for row in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("source story IDs are invalid or duplicated")
    facts = sum(len(row.get("result", {}).get("benchmark_stages", {}).get("retrieval", {}).get("items", [])) for row in rows)
    if facts != 80:
        raise ValueError(f"Benchmark 3.2A requires exactly 80 frozen facts; found {facts}")
    for row in rows:
        stages = row.get("result", {}).get("benchmark_stages", {})
        retrieval = stages.get("retrieval")
        judge = stages.get("judge")
        if not isinstance(retrieval, dict) or not isinstance(judge, dict):
            raise ValueError(f"{row.get('case_id')}: missing frozen retrieval or judge")
        if len(retrieval.get("items", [])) != len(judge.get("items", [])):
            raise ValueError(f"{row.get('case_id')}: frozen retrieval/judge item mismatch")
    return {
        "source": source, "rows": rows, "stories": 24, "facts": 80,
        "hashes": {path.name: sha256(path) for path in required},
    }
