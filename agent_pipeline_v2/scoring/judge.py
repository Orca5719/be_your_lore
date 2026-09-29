"""Judge classification metrics for oracle and pipeline-conditioned facts."""

from __future__ import annotations

from collections import Counter
from typing import Any


LABELS = ("一致", "矛盾", "不确定")
CODE_TO_LABEL = {"consistent": "一致", "contradiction": "矛盾", "uncertain": "不确定"}


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"count": numerator, "total": denominator, "value": numerator / denominator if denominator else None}


def classification_metrics(rows: list[dict], labels: tuple[str, ...] = LABELS) -> dict:
    """Return accuracy, confusion matrix, macro-F1 and per-class metrics."""
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    missing = 0
    correct = 0
    for row in rows:
        expected = row.get("expected")
        predicted = row.get("predicted")
        if expected not in labels:
            continue
        if predicted not in labels:
            missing += 1
        else:
            matrix[expected][predicted] += 1
            correct += expected == predicted
    per_class = {}
    f1_values = []
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values()) + sum(1 for row in rows if row.get("expected") == label and row.get("predicted") not in labels)
        predicted_count = sum(matrix[expected][label] for expected in labels)
        precision = tp / predicted_count if predicted_count else None
        recall = tp / support if support else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if support else None)
        per_class[label] = {"tp": tp, "support": support, "predicted": predicted_count, "precision": precision, "recall": recall, "f1": f1}
        if support:
            f1_values.append(f1 or 0.0)
    total = len([row for row in rows if row.get("expected") in labels])
    return {
        "accuracy": {"correct": correct, "total": total, "value": correct / total if total else None},
        "confusion_matrix": matrix,
        "missing_or_invalid_predictions": missing,
        "per_class": per_class,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
    }


def _events(row: dict) -> set[str]:
    values = row.get("result", {}).get("judge", {}).get("events", [])
    return {item["id"] for item in values if isinstance(item, dict) and isinstance(item.get("id"), str)}


def _items(row: dict) -> dict[str, dict]:
    values = row.get("result", {}).get("judge", {}).get("items", [])
    return {item["event_id"]: item for item in values if isinstance(item, dict) and isinstance(item.get("event_id"), str)}


def _evidence(row: dict, event_id: str) -> set[str]:
    values = row.get("result", {}).get("judge", {}).get("retrieval", {}).get("items", [])
    for item in values:
        if isinstance(item, dict) and item.get("event_id") == event_id:
            return {evidence["id"] for evidence in item.get("evidence", []) if isinstance(evidence, dict) and isinstance(evidence.get("id"), str)}
    return set()


def score_pipeline_judge(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict:
    rows = {row.get("case_id"): row for row in run_rows}
    conditioned, strict = [], []
    extraction_misses = 0
    reasoning_labels = review.get("reasoning_support_labels", {})
    unsupported = 0
    answer_total = 0
    for case in dataset_cases:
        case_id = case["id"]
        row = rows.get(case_id, {})
        event_ids = _events(row)
        items = _items(row)
        labels = review.get("system_event_labels", {}).get(case_id, {})
        mappings = review.get("event_mapping", {}).get(case_id, {})
        for gold in case.get("gold_facts", []):
            candidates = mappings.get(gold["id"], [])
            if not isinstance(candidates, list):
                candidates = [candidates]
            valid_ids = []
            for candidate in candidates:
                event_id = candidate if isinstance(candidate, str) else candidate.get("event_id") if isinstance(candidate, dict) else None
                if event_id in event_ids and labels.get(event_id, {}).get("label") == "valid_checkable" and event_id in items:
                    valid_ids.append(event_id)
            if not valid_ids:
                extraction_misses += 1
                continue
            event_id = valid_ids[0]
            item = items[event_id]
            predicted = CODE_TO_LABEL.get(item.get("verdict"))
            record = {"expected": gold.get("expected_verdict"), "predicted": predicted, "case_id": case_id, "gold_id": gold["id"], "event_id": event_id}
            conditioned.append(record)
            answer_total += 1
            support = reasoning_labels.get(case_id, {}).get(event_id, {})
            if support.get("label") == "unsupported":
                unsupported += 1
            minimum_sets = [set(group) for group in gold.get("minimum_evidence_sets", []) if group]
            evidence = _evidence(row, event_id)
            if not minimum_sets or any(group.issubset(evidence) for group in minimum_sets):
                strict.append(record)
    return {
        "schema_version": "agent-pipeline-v2-judge-metrics-v1",
        "pipeline": classification_metrics(conditioned),
        "strict_evidence": classification_metrics(strict),
        "extraction_misses": extraction_misses,
        "strict_evidence_misses": len(conditioned) - len(strict),
        "unsupported_reasoning": _ratio(unsupported, answer_total),
    }
