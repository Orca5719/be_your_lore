"""Reviewed event extraction metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _ratio(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "value": count / total if total else None}


def _event_rows(run_row: dict) -> list[dict]:
    result = run_row.get("result", {})
    judge = result.get("judge", {}) if isinstance(result, dict) else {}
    events = judge.get("events", [])
    return events if isinstance(events, list) else []


def _case_metrics(case: dict, row: dict, review: dict) -> dict:
    case_id = case["id"]
    events = _event_rows(row)
    labels = review.get("system_event_labels", {}).get(case_id, {})
    mapping = review.get("event_mapping", {}).get(case_id, {})
    event_ids = {event.get("id") for event in events}
    gold = case.get("gold_facts", [])
    gold_ids = {fact["id"] for fact in gold}
    matched = set()
    for gold_id in gold_ids:
        candidates = mapping.get(gold_id, [])
        if not isinstance(candidates, list):
            candidates = [candidates]
        for candidate in candidates:
            event_id = candidate if isinstance(candidate, str) else candidate.get("event_id") if isinstance(candidate, dict) else None
            if event_id in event_ids and labels.get(event_id, {}).get("label") == "valid_checkable":
                matched.add(gold_id)
                break
    valid = sum(labels.get(event_id, {}).get("label") == "valid_checkable" for event_id in event_ids)
    hallucinated = sum(labels.get(event_id, {}).get("label") == "hallucinated" for event_id in event_ids)
    overselected = sum(labels.get(event_id, {}).get("label") == "overselected" for event_id in event_ids)
    duplicate = sum(labels.get(event_id, {}).get("label") == "duplicate" for event_id in event_ids)
    return {
        "gold_ids": gold_ids,
        "matched_gold_ids": matched,
        "event_count": len(event_ids),
        "valid": valid,
        "hallucinated": hallucinated,
        "overselected": overselected,
        "duplicate": duplicate,
    }


def _summarize(parts: list[dict]) -> dict:
    gold = set().union(*(part["gold_ids"] for part in parts)) if parts else set()
    matched = set().union(*(part["matched_gold_ids"] for part in parts)) if parts else set()
    event_total = sum(part["event_count"] for part in parts)
    valid = sum(part["valid"] for part in parts)
    hallucinated = sum(part["hallucinated"] for part in parts)
    overselected = sum(part["overselected"] for part in parts)
    duplicate = sum(part["duplicate"] for part in parts)
    return {
        "recall": {"hits": len(matched), "total": len(gold), "value": len(matched) / len(gold) if gold else None},
        "precision": {"hits": valid, "total": event_total, "value": valid / event_total if event_total else None},
        "hallucination_rate": _ratio(hallucinated, event_total),
        "overselection_rate": _ratio(overselected, event_total),
        "duplicate_rate": _ratio(duplicate, event_total),
        "emitted_events": event_total,
    }


def score_extraction(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict:
    """Score reviewed extraction outputs, keeping gold and emitted denominators explicit."""
    rows = {row.get("case_id"): row for row in run_rows}
    parts_by_case: dict[str, dict] = {}
    for case in dataset_cases:
        parts_by_case[case["id"]] = _case_metrics(case, rows.get(case["id"], {}), review)
    overall = _summarize(list(parts_by_case.values()))

    by_group: dict[str, dict] = {}
    group_parts: dict[str, list[dict]] = defaultdict(list)
    for case in dataset_cases:
        group_parts[case.get("group", "unknown")].append(parts_by_case[case["id"]])
    for group, parts in group_parts.items():
        by_group[group] = _summarize(parts)

    by_dimension: dict[str, dict] = {}
    dimension_parts: dict[str, list[dict]] = defaultdict(list)
    for case in dataset_cases:
        dimensions = {fact.get("dimension") for fact in case.get("gold_facts", []) if fact.get("dimension")}
        for dimension in dimensions:
            dimension_parts[dimension].append(parts_by_case[case["id"]])
    for dimension, parts in dimension_parts.items():
        by_dimension[dimension] = _summarize(parts)
    return {
        "schema_version": "agent-pipeline-v2-extraction-metrics-v1",
        **overall,
        "by_group": by_group,
        "by_dimension": by_dimension,
    }
