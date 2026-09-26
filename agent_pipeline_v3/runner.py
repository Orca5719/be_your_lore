from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import uuid

from .schema import stable_digest, validate_run_row


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(row.get("case_id"), row.get("system")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate run rows")
    return rows


def build_manifest(*, dataset: dict, configs: list[dict], inputs: dict[str, str], repeats: int, warmup: int) -> dict:
    identity = {"dataset": stable_digest(dataset), "configs": configs, "inputs": inputs, "repeats": repeats, "warmup": warmup}
    return {
        "schema_version": "agent-pipeline-v3-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": stable_digest(identity),
    }


def ensure_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("identity_sha256") != manifest.get("identity_sha256"):
            raise ValueError("resume configuration hash mismatch")
    else:
        atomic_write_json(path, manifest)


def run_resumable(cases: list[dict], systems: list, output: Path, process, *, on_start=None, progress_factory=None) -> list[dict]:
    rows = load_jsonl(output)
    done = {(row["case_id"], row["system"]) for row in rows}
    for case in cases:
        for system in systems:
            key = (case["id"], system.config.name)
            if key in done:
                continue
            if on_start:
                on_start(case, system, len(rows), len(cases) * len(systems))
            if progress_factory:
                row = process(system, case["id"], case["story"], progress=progress_factory(case, system))
            else:
                row = process(system, case["id"], case["story"])
            validate_run_row(row)
            rows.append(row)
            done.add(key)
            write_jsonl_atomic(output, rows)
    return rows


def aggregate_performance(repeat_rows: list[list[dict]], warmup_count: int = 1) -> dict:
    measured = repeat_rows[warmup_count:]
    totals = [sum(row["stage_metrics"]["total"]["seconds"] for row in run) for run in measured]
    facts = [sum(len(row.get("result", {}).get("judge", {}).get("items", [])) for row in run) for run in measured]
    stories = [len(run) for run in measured]
    def median(values):
        return statistics.median(values) if values else None
    token_totals = []
    for run in measured:
        total = {"input_tokens": 0, "generated_tokens": 0}
        for row in run:
            judge = row.get("result", {}).get("judge", {})
            metrics = judge.get("metrics", {}) if isinstance(judge, dict) else {}
            total["input_tokens"] += int(metrics.get("useful_input_tokens", 0) or 0)
            total["generated_tokens"] += int(metrics.get("generated_tokens", 0) or 0)
            for report in judge.get("batch_reports", []) if isinstance(judge, dict) else []:
                for call in report.get("batch_calls", []):
                    timing = call.get("timing", {})
                    total["input_tokens"] += int(timing.get("useful_input_tokens", 0) or 0)
                    total["generated_tokens"] += int(timing.get("total_generated_tokens", 0) or 0)
        token_totals.append(total)
    return {
        "warmup_runs": warmup_count,
        "measured_runs": len(measured),
        "median_total_seconds": median(totals),
        "median_stories_per_second": median([n / t for n, t in zip(stories, totals) if t]),
        "median_facts_per_second": median([n / t for n, t in zip(facts, totals) if t]),
        "peak_allocated_bytes": max((row.get("cuda", {}).get("peak_allocated_bytes", 0) for run in measured for row in run), default=0),
        "peak_reserved_bytes": max((row.get("cuda", {}).get("peak_reserved_bytes", 0) for run in measured for row in run), default=0),
        "median_input_tokens": median([value["input_tokens"] for value in token_totals]),
        "median_generated_tokens": median([value["generated_tokens"] for value in token_totals]),
        "status_counts": dict(Counter(row["status"] for run in measured for row in run)),
    }
