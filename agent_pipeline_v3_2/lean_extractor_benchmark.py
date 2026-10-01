from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import time

from agent_pipeline_v2.scoring.extraction import score_extraction
from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
from agent_pipeline_v3.metrics import aggregate_profile, enrich_story_calls

from .extractor_audit import aggregate_audit, build_audit_rows
from .lean_extractor import extract_events_lean

EVENT_SEMANTIC_FIELDS = ("actors", "event", "mental_state", "explicit", "modality", "conditions", "source_ids", "context_ids", "check_reason")


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
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


def _semantic_key(event: dict) -> str:
    return json.dumps({field: event.get(field) for field in EVENT_SEMANTIC_FIELDS}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candidate_review(previous: dict) -> dict:
    sections = ("event_mapping", "system_event_labels")
    result = {section: {} for section in sections}
    for section in sections:
        for key, value in previous.get(section, {}).items():
            if key.startswith("candidate:"):
                result[section][key[len("candidate:"):]] = value
    return result


def reuse_exact_event_reviews(dataset: dict, source_rows: list[dict], lean_rows: list[dict], previous: dict) -> dict:
    old_rows = {row["case_id"]: row for row in source_rows}
    old_review = candidate_review(previous)
    ledger = {"status": "reviewed", "provenance": "assistant-reviewed", "event_mapping": {}, "system_event_labels": {}, "pending": []}
    cases = {case["id"]: case for case in dataset["cases"]}
    for row in lean_rows:
        case_id = row["case_id"]
        old_events = old_rows[case_id]["result"]["benchmark_stages"]["extraction"].get("events", [])
        by_key: dict[str, list[dict]] = {}
        for event in old_events:
            by_key.setdefault(_semantic_key(event), []).append(event)
        old_labels = old_review["system_event_labels"].get(case_id, {})
        old_mapping = old_review["event_mapping"].get(case_id, {})
        ledger["event_mapping"][case_id] = {gold["id"]: [] for gold in cases[case_id].get("gold_facts", [])}
        ledger["system_event_labels"][case_id] = {}
        for event in row["result"]["benchmark_stages"]["extraction"].get("events", []):
            matches = by_key.get(_semantic_key(event), [])
            if len(matches) == 1 and matches[0].get("id") in old_labels:
                old_id = matches[0]["id"]
                label = copy.deepcopy(old_labels[old_id])
                label["reused_from_event_id"] = old_id
                ledger["system_event_labels"][case_id][event["id"]] = label
                for gold_id, event_ids in old_mapping.items():
                    if old_id in event_ids:
                        ledger["event_mapping"][case_id].setdefault(gold_id, []).append(event["id"])
            else:
                ledger["system_event_labels"][case_id][event["id"]] = {"label": "pending", "gold_ids": [], "note": "Lean Extractor event differs from frozen reviewed event"}
                ledger["pending"].append({"case_id": case_id, "event_id": event["id"], "event": event})
    if ledger["pending"]:
        ledger["status"] = "pending_review"
        ledger["provenance"] = "unreviewed"
    ledger["pending_event_count"] = len(ledger["pending"])
    ledger["reused_event_count"] = sum(value.get("label") != "pending" for case in ledger["system_event_labels"].values() for value in case.values())
    return ledger


def build_extraction_row(case_id: str, extraction: dict, raw_calls: list[dict], elapsed: float) -> dict:
    result = {
        "judge": {"events": copy.deepcopy(extraction["events"]), "items": []},
        "benchmark_stages": {"extraction": extraction, "judge": {"batch_reports": [], "items": []}},
    }
    row = {
        "case_id": case_id, "system": "lean-extractor", "status": extraction["status"], "result": result,
        "stage_metrics": {"extraction": {"seconds": elapsed}, "retrieval": {"seconds": 0.0}, "judge": {"seconds": 0.0}, "report": {"seconds": 0.0}, "total": {"seconds": elapsed}},
    }
    row["calls"] = enrich_story_calls(row, raw_calls)
    return row


def run_extraction_cases(cases: list[dict], llm, output: Path, progress=None) -> list[dict]:
    path = output / "extractor_runs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    prior_calls = [call for row in rows for call in row.get("calls", [])]
    llm.collector.resume_after(prior_calls)
    completed = {row["case_id"] for row in rows}
    repaired = ExtractionRepairLLM(llm)
    for case in cases:
        if case["id"] in completed:
            continue
        llm.set_story_id(case["id"])
        before = len(llm.collector.records)
        started = time.perf_counter()
        extraction = extract_events_lean(case["story"], device=llm.device, llm=repaired)
        elapsed = time.perf_counter() - started
        rows.append(build_extraction_row(case["id"], extraction, llm.collector.records[before:], elapsed))
        _atomic_jsonl(path, rows)
        if progress:
            progress(len(rows), len(cases), case["id"])
    return rows


def summarize_extractor(*, dataset: dict, source_rows: list[dict], lean_rows: list[dict], previous_review: dict, baseline_profile: dict, model_load_seconds: float) -> dict:
    calls = [call for row in lean_rows for call in row.get("calls", [])]
    profile = aggregate_profile(calls, lean_rows, model_load_seconds)
    review = reuse_exact_event_reviews(dataset, source_rows, lean_rows, previous_review)
    baseline_review = candidate_review(previous_review)
    baseline_quality = score_extraction(dataset["cases"], source_rows, baseline_review)
    lean_quality = score_extraction(dataset["cases"], lean_rows, review)
    audit = aggregate_audit(build_audit_rows(lean_rows, calls))
    uncovered = sum(len(row["result"]["benchmark_stages"]["extraction"].get("uncovered_span_ids", [])) for row in lean_rows)
    baseline = baseline_profile["components"]["extractor"]
    current = profile["components"]["extractor"]
    quality_complete = review["pending_event_count"] == 0
    baseline_hits = baseline_quality["recall"]["hits"]
    lean_hits = lean_quality["recall"]["hits"]
    baseline_hallucinated = baseline_quality["hallucination_rate"]["count"]
    lean_hallucinated = lean_quality["hallucination_rate"]["count"]
    gate = {
        "status": "pass" if quality_complete and lean_hits >= baseline_hits - 1 and uncovered == 0 and lean_hallucinated <= baseline_hallucinated else "pending_review" if not quality_complete else "fail",
        "quality_complete": quality_complete,
        "recall_within_one_gold": lean_hits >= baseline_hits - 1 if quality_complete else None,
        "no_uncovered_spans": uncovered == 0,
        "hallucinations_not_increased": lean_hallucinated <= baseline_hallucinated if quality_complete else None,
    }
    total_seconds = profile["workflow"]["end_to_end_seconds"]
    return {
        "schema_version": "agent-pipeline-v3.2-lean-extractor-summary-v1",
        "stories": len(lean_rows), "emitted_events": sum(len(row["result"]["benchmark_stages"]["extraction"].get("events", [])) for row in lean_rows),
        "uncovered_span_count": uncovered, "review": review, "quality_gate": gate,
        "quality": {"baseline": baseline_quality, "lean": lean_quality, "formal": quality_complete},
        "performance": {"baseline": baseline, "lean": current, "profile": profile,
            "stories_per_second": len(lean_rows) / total_seconds if total_seconds else None,
            "events_per_second": sum(len(row["result"]["benchmark_stages"]["extraction"].get("events", [])) for row in lean_rows) / total_seconds if total_seconds else None,
            "output_token_change_ratio": (current["output_tokens"] - baseline["output_tokens"]) / baseline["output_tokens"],
            "decode_change_ratio": (current["decode_seconds"] - baseline["decode_seconds"]) / baseline["decode_seconds"],
            "total_change_ratio": (current["total_seconds"] - baseline["total_seconds"]) / baseline["total_seconds"],
        },
        "failure_audit": audit,
    }


def write_extractor_outputs(output: Path, summary: dict) -> None:
    (output / "extractor_lean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "lean_extractor_review.json").write_text(json.dumps(summary["review"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    perf, quality, audit = summary["performance"], summary["quality"], summary["failure_audit"]
    b, n = perf["baseline"], perf["lean"]
    lines = [
        "# Benchmark 3.2C — Lean Extractor", "",
        f"- Stories: {summary['stories']}", f"- Emitted events: {summary['emitted_events']}",
        f"- Uncovered spans: {summary['uncovered_span_count']}",
        f"- Exact reviews reused: {summary['review']['reused_event_count']}",
        f"- Pending event reviews: {summary['review']['pending_event_count']}",
        f"- Quality gate: **{summary['quality_gate']['status']}**", "",
        "## Performance", "", "| Metric | Benchmark 3 | Lean Extractor |", "|---|---:|---:|",
        f"| Calls | {b['calls']} | {n['calls']} |", f"| Retry calls | {b['retry_calls']} | {n['retry_calls']} |",
        f"| Output tokens | {b['output_tokens']} | {n['output_tokens']} |",
        f"| Prefill s | {b['prefill_seconds']:.3f} | {n['prefill_seconds']:.3f} |",
        f"| Decode s | {b['decode_seconds']:.3f} | {n['decode_seconds']:.3f} |",
        f"| LLM total s | {b['total_seconds']:.3f} | {n['total_seconds']:.3f} |",
        f"| TTFT p50 ms | {b['ttft_p50_ms']:.3f} | {n['ttft_p50_ms']:.3f} |",
        f"| Peak allocated | {b['peak_allocated']} | {n['peak_allocated']} |", "",
        "## Extraction quality", "", "| Metric | Benchmark 3 | Lean Extractor |", "|---|---:|---:|",
        f"| Recall | {quality['baseline']['recall']['value']} | {quality['lean']['recall']['value']} |",
        f"| Hallucinated events | {quality['baseline']['hallucination_rate']['count']} | {quality['lean']['hallucination_rate']['count']} |",
        "",
        "Quality numbers are formal only when pending event reviews equal zero.", "",
        "## Failure cost", "", "| Class | Calls | Output tok | Total s |", "|---|---:|---:|---:|",
    ]
    for name, item in audit["cost_classes"].items():
        lines.append(f"| {name} | {item['calls']} | {item['output_tokens']} | {item['total_time']:.3f} |")
    (output / "extractor_lean_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
