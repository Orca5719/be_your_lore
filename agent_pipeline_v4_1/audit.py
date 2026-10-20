"""Independent extraction and end-to-end quality accounting."""

from __future__ import annotations

from collections import Counter
import hashlib
import json


LABELS = {"valid_checkable", "hallucinated", "overselected", "duplicate"}
POSITIVE = "contradiction"


def event_signature(event: dict) -> str:
    fields = ("actors", "event", "mental_state", "explicit", "modality", "conditions",
              "source_ids", "context_ids", "check_reason")
    payload = json.dumps({name: event.get(name) for name in fields},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _events(row: dict) -> list[dict]:
    return row.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("events", [])


def validate_labels(cases: list[dict], rows: list[dict], labels: dict) -> None:
    expected = {case["id"] for case in cases}
    if len(rows) != len(expected) or {row["case_id"] for row in rows} != expected:
        raise ValueError("expected one extraction result for each of 24 stories")
    case_map = {case["id"]: case for case in cases}
    for row in rows:
        case_id = row["case_id"]
        gold = {fact["id"] for fact in case_map[case_id]["gold_facts"]}
        actual_ids = {event["id"] for event in _events(row)}
        if len(actual_ids) != len(_events(row)) or set(labels.get(case_id, {})) != actual_ids:
            raise ValueError(f"missing, extra or duplicate event labels: {case_id}")
        for event in _events(row):
            value = labels[case_id][event["id"]]
            if value.get("signature") != event_signature(event):
                raise ValueError(f"stale review signature: {case_id}/{event['id']}")
            if value.get("label") not in LABELS:
                raise ValueError(f"event not reviewed: {case_id}/{event['id']}")
            mapped = value.get("gold_ids")
            if not isinstance(mapped, list) or len(mapped) != len(set(mapped)) or not set(mapped) <= gold:
                raise ValueError(f"invalid gold mapping: {case_id}/{event['id']}")
            if value["label"] != "valid_checkable" and mapped:
                raise ValueError(f"non-valid event cannot map gold: {case_id}/{event['id']}")


def score_extraction(cases: list[dict], rows: list[dict], labels: dict) -> dict:
    validate_labels(cases, rows, labels)
    counts = Counter()
    misses = []
    for case in cases:
        case_id = case["id"]
        values = labels[case_id].values()
        counts.update(value["label"] for value in values)
        found = {gold_id for value in values for gold_id in value["gold_ids"]}
        misses.extend(f"{case_id}/{gold['id']}" for gold in case["gold_facts"] if gold["id"] not in found)
    total = sum(counts.values())
    gold_total = sum(len(case["gold_facts"]) for case in cases)
    hits = gold_total - len(misses)
    return {
        "gold_hits": hits, "gold_total": gold_total, "recall": hits / gold_total if gold_total else None,
        "valid_events": counts["valid_checkable"], "emitted_events": total,
        "precision": counts["valid_checkable"] / total if total else None,
        "hallucinated_events": counts["hallucinated"], "hallucination_rate": counts["hallucinated"] / total if total else None,
        "overselected_events": counts["overselected"], "overselect_rate": counts["overselected"] / total if total else None,
        "duplicate_events": counts["duplicate"], "missed_gold": misses,
    }


def score_end_to_end(cases: list[dict], rows: list[dict], labels: dict) -> dict:
    case_map = {case["id"]: case for case in cases}
    if len(rows) != len(cases) or {row["case_id"] for row in rows} != set(case_map):
        raise ValueError("end-to-end rows do not cover exactly the dataset")
    tp = fp = 0
    misses = []
    false_positives = []
    for row in rows:
        cid = row["case_id"]
        expected = {gold["id"] for gold in case_map[cid]["gold_facts"]
                    if gold["expected_verdict"] == "矛盾"}
        matched = set()
        items = row.get("result", {}).get("judge", {}).get("items", [])
        for item in items:
            if item.get("status") != "ok" or item.get("verdict") != POSITIVE:
                continue
            eid = item["event_id"]
            value = labels[cid].get(eid)
            if not value:
                raise ValueError(f"judge event missing review: {cid}/{eid}")
            available = (set(value["gold_ids"]) & expected) - matched if value["label"] == "valid_checkable" else set()
            if available:
                matched.add(sorted(available)[0])
                tp += 1
            else:
                fp += 1
                false_positives.append(f"{cid}/{eid}")
        misses.extend(f"{cid}/{gold_id}" for gold_id in sorted(expected - matched))
    fn = len(misses)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "missed_gold_contradictions": misses, "false_positive_events": false_positives}
