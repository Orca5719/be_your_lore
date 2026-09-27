from __future__ import annotations

import math
import statistics

from .profiling import COMPONENTS, validate_call_record


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _sum(rows: list[dict], field: str) -> float:
    return sum(float(row.get(field, 0) or 0) for row in rows)


def _maximum(rows: list[dict], field: str):
    values = [row[field] for row in rows if row.get(field) is not None]
    return max(values) if values else None


def _component_metrics(rows: list[dict], total_llm_seconds: float) -> dict:
    prefill = _sum(rows, "prefill_time")
    decode = _sum(rows, "decode_time")
    total = _sum(rows, "total_time")
    inputs = sum(int(row["input_tokens"]) for row in rows)
    padding = sum(int(row["padded_input_tokens"]) for row in rows)
    outputs = sum(int(row["output_tokens"]) for row in rows)
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    return {
        "calls": len(rows),
        "retry_calls": sum(bool(row.get("is_retry")) for row in rows),
        "avg_batch_size": statistics.mean(row["batch_size"] for row in rows) if rows else None,
        "input_tokens": inputs,
        "padded_input_tokens": padding,
        "compute_input_tokens": inputs + padding,
        "output_tokens": outputs,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "total_seconds": total,
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
        "ttft_p95_ms": _nearest_rank(ttfts, 0.95),
        "ttft_max_ms": max(ttfts) if ttfts else None,
        "prefill_useful_tokens_per_second": inputs / prefill if prefill else None,
        "prefill_compute_tokens_per_second": (inputs + padding) / prefill if prefill else None,
        "decode_tokens_per_second": outputs / decode if decode else None,
        "peak_allocated": _maximum(rows, "peak_allocated"),
        "peak_reserved": _maximum(rows, "peak_reserved"),
        "llm_time_share": total / total_llm_seconds if total_llm_seconds else None,
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in ("ok", "error")},
    }


def enrich_story_calls(story_row: dict, calls: list[dict]) -> list[dict]:
    stages = story_row.get("result", {}).get("benchmark_stages", {})
    extraction = stages.get("extraction", {})
    judge = stages.get("judge", {})
    metadata: dict[str, dict] = {}
    for window in extraction.get("calls", []):
        for attempt in window.get("attempts", []):
            call_id = attempt.get("timing", {}).get("call_id")
            if not call_id:
                continue
            metadata[call_id] = {
                "attempt": attempt.get("attempt"),
                "operation": attempt.get("purpose", window.get("purpose", "extract_checkable_events")),
                "window_id": window.get("window_id"),
                "request_ids": list(window.get("target_ids", [])),
                "is_retry": int(attempt.get("attempt", 1) or 1) > 1,
            }
    for batch_report in judge.get("batch_reports", []):
        for batch_call in batch_report.get("batch_calls", []):
            call_id = batch_call.get("timing", {}).get("call_id")
            if not call_id:
                continue
            attempt = int(batch_call.get("attempt", 1) or 1)
            metadata[call_id] = {
                "attempt": attempt,
                "operation": "judge_retry" if attempt > 1 else "batched_judge",
                "window_id": None,
                "request_ids": list(batch_call.get("request_ids", [])),
                "is_retry": attempt > 1,
            }
    enriched = []
    for original in calls:
        validate_call_record(original)
        call_id = original["call_id"]
        if call_id not in metadata:
            raise ValueError(f"trace {call_id} is not linked to an extraction or judge call")
        enriched.append({**original, **metadata[call_id]})
    return enriched


def aggregate_profile(calls: list[dict], story_rows: list[dict], model_load_seconds: float) -> dict:
    for call in calls:
        validate_call_record(call)
    total_llm = _sum(calls, "total_time")
    components = {
        component: _component_metrics([row for row in calls if row["component"] == component], total_llm)
        for component in sorted(COMPONENTS)
    }
    stage_names = ("extraction", "retrieval", "judge", "report", "total")
    stage_seconds = {
        stage: sum(float(row.get("stage_metrics", {}).get(stage, {}).get("seconds", 0) or 0) for row in story_rows)
        for stage in stage_names
    }
    end_to_end = stage_seconds["total"]
    non_llm = end_to_end - total_llm
    if non_llm < -1e-6:
        raise ValueError("LLM generation time exceeds end-to-end time")
    non_llm = max(0.0, non_llm)
    facts = sum(len(row.get("result", {}).get("judge", {}).get("items", [])) for row in story_rows)
    stories = len(story_rows)
    retry_rows = [row for row in calls if row.get("is_retry")]
    workflow = {
        "model_load_seconds": float(model_load_seconds),
        "stage_seconds": stage_seconds,
        "llm_generation_seconds": total_llm,
        "non_llm_seconds": non_llm,
        "end_to_end_seconds": end_to_end,
        "story_count": stories,
        "fact_count": facts,
        "stories_per_second": stories / end_to_end if end_to_end else None,
        "facts_per_second": facts / end_to_end if end_to_end else None,
        "calls_per_story": len(calls) / stories if stories else None,
        "tokens_per_story": sum(row["input_tokens"] + row["output_tokens"] for row in calls) / stories if stories else None,
        "report_llm_calls": 0,
    }
    component_seconds = sum(value["total_seconds"] for value in components.values())
    explained_seconds = total_llm + non_llm
    return {
        "schema_version": "agent-pipeline-v3-profile-summary-v1",
        "components": components,
        "workflow": workflow,
        "retry_cost": {
            "calls": len(retry_rows),
            "input_tokens": sum(row["input_tokens"] for row in retry_rows),
            "output_tokens": sum(row["output_tokens"] for row in retry_rows),
            "total_time": _sum(retry_rows, "total_time"),
        },
        "slowest_calls": sorted(calls, key=lambda row: row["total_time"], reverse=True)[:10],
        "alignment": {
            "component_llm_seconds": component_seconds,
            "call_llm_seconds": total_llm,
            "component_call_delta_seconds": component_seconds - total_llm,
            "explained_end_to_end_seconds": explained_seconds,
            "end_to_end_delta_seconds": explained_seconds - end_to_end,
        },
    }
