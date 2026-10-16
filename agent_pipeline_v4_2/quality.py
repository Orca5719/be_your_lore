"""Review ledger for changed Extractor events and downstream scoring."""

from __future__ import annotations

from agent_pipeline_v4_1.audit import event_signature, score_end_to_end, score_extraction, validate_labels
from agent_pipeline_v2_2.review import _event_signature as frozen_event_signature


def _events(row: dict) -> list[dict]:
    return row["result"]["benchmark_stages"]["extraction"]["events"]


def build_review_template(source_rows: list[dict], candidate_rows: list[dict], old_review: dict, mode: str) -> dict:
    old_rows = {row["case_id"]: row for row in source_rows}
    cases = {}
    pending = []
    for row in candidate_rows:
        case_id = row["case_id"]
        source = old_rows[case_id]
        by_signature: dict[str, list[dict]] = {}
        for event in _events(source):
            by_signature.setdefault(event_signature(event), []).append(event)
        labels = old_review["system_event_labels"].get("candidate:" + case_id, {})
        case_labels = {}
        for event in _events(row):
            signature = event_signature(event)
            matches = by_signature.get(signature, [])
            match = matches[0] if len(matches) == 1 else None
            previous = labels.get(match["id"]) if match else None
            if previous is not None and previous.get("signature") != frozen_event_signature(case_id, "candidate", match):
                previous = None
            if previous is None:
                case_labels[event["id"]] = {"signature": signature, "label": None,
                                             "gold_ids": [], "note": None}
                pending.append(case_id + "/" + event["id"])
            else:
                case_labels[event["id"]] = {"signature": signature, "label": previous["label"],
                                             "gold_ids": list(previous.get("gold_ids", [])),
                                             "note": "与冻结基线事件语义字段相同；沿用原审核"}
        cases[case_id] = case_labels
    return {"schema_version": "benchmark-4.2A-review-v1", "mode": mode,
            "provenance": "baseline-label-reuse-plus-pending-review", "cases": cases,
            "pending": pending}


def score_reviewed(cases: list[dict], extraction_rows: list[dict], downstream_rows: list[dict], review: dict) -> dict:
    if review.get("pending"):
        raise ValueError("仍有待审核事件，不能发布质量分数")
    labels = review["cases"]
    validate_labels(cases, extraction_rows, labels)
    return {"extraction": score_extraction(cases, extraction_rows, labels),
            "end_to_end": score_end_to_end(cases, downstream_rows, labels)}
