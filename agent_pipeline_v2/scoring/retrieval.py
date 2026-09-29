"""Reviewed retrieval quality metrics for the story pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _value(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _event_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("event_id")
    return None


def _result_events(row: dict) -> set[str]:
    events = row.get("result", {}).get("judge", {}).get("events", [])
    return {event.get("id") for event in events if isinstance(event, dict) and isinstance(event.get("id"), str)}


def _retrieval_items(row: dict) -> dict[str, dict]:
    items = row.get("result", {}).get("judge", {}).get("retrieval", {}).get("items", [])
    return {
        item["event_id"]: item
        for item in items
        if isinstance(item, dict) and isinstance(item.get("event_id"), str)
    }


def _chunk_ids(item: dict | None, k: int) -> list[str]:
    if not item or item.get("status", "ok") != "ok":
        return []
    result = []
    for evidence in item.get("evidence", [])[:k]:
        if isinstance(evidence, dict) and isinstance(evidence.get("id"), str):
            result.append(evidence["id"])
    return result


def _override_ids(review: dict, case_id: str, event_id: str, relevant: set[str]) -> set[str]:
    row = review.get("retrieval_relevance_overrides", {}).get(case_id, {}).get(event_id, {})
    if not isinstance(row, dict):
        return relevant
    result = set(relevant)
    result.update(value for value in row.get("relevant_ids", []) if isinstance(value, str))
    result.difference_update(value for value in row.get("irrelevant_ids", []) if isinstance(value, str))
    return result


def _summarize(gold_records: list[dict], event_records: list[dict], missing: int) -> dict:
    eligible = [record for record in gold_records if record["requires_evidence"]]
    recall_hits = sum(record["hit"] for record in eligible)
    reciprocal_sum = sum(record["reciprocal_rank"] for record in eligible)
    relevant = sum(record["relevant"] for record in event_records)
    returned = sum(record["returned"] for record in event_records)
    irrelevant = returned - relevant
    return {
        "recall_at_k": {"hits": recall_hits, "total": len(eligible), "value": _value(recall_hits, len(eligible))},
        "mrr": {"sum_reciprocal_rank": reciprocal_sum, "total": len(eligible), "value": _value(reciprocal_sum, len(eligible))},
        "precision_at_k": {"relevant": relevant, "total": returned, "value": _value(relevant, returned)},
        "noise_rate_at_k": {"irrelevant": irrelevant, "total": returned, "value": _value(irrelevant, returned)},
        "event_diagnostics": {
            "scored_events": len(event_records),
            "missing_retrieval_events": missing,
            "returned_chunks": returned,
        },
    }


def score_retrieval(dataset_cases: list[dict], run_rows: list[dict], review: dict, *, k: int = 5) -> dict:
    """Score gold-level evidence recall/MRR and event-level retrieval noise."""
    if type(k) is not int or k < 1:
        raise ValueError("k必须为正整数")
    rows = {row.get("case_id"): row for row in run_rows}
    all_gold_records = []
    all_event_records = []
    missing_total = 0
    gold_by_dimension: dict[str, list[dict]] = defaultdict(list)
    events_by_dimension: dict[str, list[dict]] = defaultdict(list)
    missing_by_dimension: dict[str, int] = defaultdict(int)

    for case in dataset_cases:
        case_id = case["id"]
        row = rows.get(case_id, {})
        available_events = _result_events(row)
        retrieval = _retrieval_items(row)
        labels = review.get("system_event_labels", {}).get(case_id, {})
        mappings = review.get("event_mapping", {}).get(case_id, {})
        facts = {fact["id"]: fact for fact in case.get("gold_facts", [])}
        valid_events = {
            event_id for event_id in available_events
            if labels.get(event_id, {}).get("label") == "valid_checkable"
        }
        gold_to_events: dict[str, list[str]] = {}
        event_to_gold: dict[str, set[str]] = defaultdict(set)
        for gold_id, candidates in mappings.items():
            if gold_id not in facts or not isinstance(candidates, list):
                continue
            event_ids = []
            for candidate in candidates:
                event_id = _event_id(candidate)
                if event_id in valid_events:
                    event_ids.append(event_id)
                    event_to_gold[event_id].add(gold_id)
            gold_to_events[gold_id] = list(dict.fromkeys(event_ids))

        for gold_id, event_ids in gold_to_events.items():
            if not event_ids:
                continue
            fact = facts[gold_id]
            relevant = set(fact.get("relevant_lore_ids", []))
            minimum_sets = [set(group) for group in fact.get("minimum_evidence_sets", []) if group]
            best_hit = False
            best_rr = 0.0
            for event_id in event_ids:
                chunks = _chunk_ids(retrieval.get(event_id), k)
                chunk_set = set(chunks)
                best_hit = best_hit or any(group.issubset(chunk_set) for group in minimum_sets)
                ranks = [index + 1 for index, chunk_id in enumerate(chunks) if chunk_id in relevant]
                if ranks:
                    best_rr = max(best_rr, 1 / min(ranks))
            record = {
                "case_id": case_id,
                "gold_id": gold_id,
                "dimension": fact.get("dimension", "unknown"),
                "requires_evidence": bool(minimum_sets),
                "hit": int(best_hit),
                "reciprocal_rank": best_rr,
            }
            all_gold_records.append(record)
            gold_by_dimension[record["dimension"]].append(record)

        for event_id, gold_ids in event_to_gold.items():
            item = retrieval.get(event_id)
            dimensions = {facts[gold_id].get("dimension", "unknown") for gold_id in gold_ids}
            relevant = {chunk_id for gold_id in gold_ids for chunk_id in facts[gold_id].get("relevant_lore_ids", [])}
            relevant = _override_ids(review, case_id, event_id, relevant)
            if item is None:
                missing_total += 1
                for dimension in dimensions:
                    missing_by_dimension[dimension] += 1
                continue
            chunks = _chunk_ids(item, k)
            record = {
                "case_id": case_id,
                "event_id": event_id,
                "dimensions": dimensions,
                "returned": len(chunks),
                "relevant": sum(chunk_id in relevant for chunk_id in chunks),
            }
            all_event_records.append(record)
            for dimension in dimensions:
                events_by_dimension[dimension].append(record)

    overall = _summarize(all_gold_records, all_event_records, missing_total)
    dimensions = set(gold_by_dimension) | set(events_by_dimension) | set(missing_by_dimension)
    by_dimension = {
        dimension: _summarize(gold_by_dimension[dimension], events_by_dimension[dimension], missing_by_dimension[dimension])
        for dimension in sorted(dimensions)
    }
    return {
        "schema_version": "agent-pipeline-v2-retrieval-metrics-v1",
        "k": k,
        **overall,
        "by_dimension": by_dimension,
    }
