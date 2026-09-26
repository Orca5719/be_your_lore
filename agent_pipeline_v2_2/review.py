from __future__ import annotations

from copy import deepcopy

from .schema import stable_digest


EVENT_LABELS = {"valid_checkable", "overselected", "hallucinated", "duplicate"}
FINDING_LABELS = {"true_positive", "false_positive", "duplicate", "unsupported"}
REASONING_LABELS = {"supported", "unsupported"}


def _events(row):
    return row.get("result", {}).get("judge", {}).get("events", [])


def _findings(row):
    return row.get("result", {}).get("findings", [])


def _event_signature(case_id: str, system: str, event: dict) -> str:
    body = {key: event.get(key) for key in ("id", "actors", "event", "mental_state", "explicit", "modality", "conditions", "source_ids", "context_ids")}
    return stable_digest({"case_id": case_id, "system": system, "event": body})


def _finding_signature(case_id: str, system: str, finding: dict) -> str:
    body = {key: finding.get(key) for key in ("id", "event_ids", "actors", "event", "verdict", "reason", "citations")}
    return stable_digest({"case_id": case_id, "system": system, "finding": body})


def build_review_template(dataset: dict, run_rows: list[dict], *, track: str = "end_to_end") -> dict:
    cases = {case["id"]: case for case in dataset["cases"]}
    systems = sorted({row["system"] for row in run_rows})
    ledger = {
        "schema_version": "agent-pipeline-v2-2-review-v1", "status": "pending_review",
        "provenance": "unreviewed", "track": track, "dataset_sha256": stable_digest(dataset),
        "systems": systems, "event_mapping": {}, "finding_mapping": {},
        "system_event_labels": {}, "system_finding_labels": {}, "reasoning_support_labels": {},
    }
    for row in run_rows:
        case_id, system = row["case_id"], row["system"]
        if case_id not in cases:
            raise ValueError(f"unknown case: {case_id}")
        key = f"{system}:{case_id}"
        ledger["event_mapping"][key] = {gold["id"]: [] for gold in cases[case_id].get("gold_facts", [])}
        ledger["finding_mapping"][key] = {gold["id"]: [] for gold in cases[case_id].get("gold_findings", [])}
        ledger["system_event_labels"][key] = {event["id"]: {"label": "pending", "gold_ids": [], "signature": _event_signature(case_id, system, event), "note": ""} for event in _events(row)}
        ledger["system_finding_labels"][key] = {finding["id"]: {"label": "pending", "gold_finding_ids": [], "signature": _finding_signature(case_id, system, finding), "note": ""} for finding in _findings(row)}
        ledger["reasoning_support_labels"][key] = {finding["id"]: {"label": "pending", "note": ""} for finding in _findings(row)}
    return ledger


def propose_review(dataset: dict, run_rows: list[dict], *, track: str = "end_to_end") -> dict:
    return build_review_template(dataset, run_rows, track=track)


def reuse_exact_reviews(template: dict, previous: dict) -> dict:
    result = deepcopy(template)
    for section in ("system_event_labels", "system_finding_labels"):
        for key, values in result[section].items():
            old_values = previous.get(section, {}).get(key, {})
            for object_id, value in values.items():
                old = old_values.get(object_id)
                if old and old.get("signature") == value.get("signature") and old.get("label") != "pending":
                    values[object_id] = deepcopy(old)
    for key, values in result["reasoning_support_labels"].items():
        old_values = previous.get("reasoning_support_labels", {}).get(key, {})
        for finding_id in values:
            signature = result["system_finding_labels"][key][finding_id]["signature"]
            old_signature = previous.get("system_finding_labels", {}).get(key, {}).get(finding_id, {}).get("signature")
            if signature == old_signature and old_values.get(finding_id, {}).get("label") in REASONING_LABELS:
                values[finding_id] = deepcopy(old_values[finding_id])
    return result


def validate_review(dataset: dict, run_rows: list[dict], review: dict, *, require_complete: bool = True) -> dict:
    if review.get("schema_version") != "agent-pipeline-v2-2-review-v1":
        raise ValueError("invalid review schema")
    if require_complete and review.get("status") != "reviewed":
        raise ValueError("review is not complete")
    if review.get("provenance") not in {"unreviewed", "assistant-reviewed", "user-reviewed"}:
        raise ValueError("invalid review provenance")
    cases = {case["id"]: case for case in dataset["cases"]}
    for row in run_rows:
        key = f'{row["system"]}:{row["case_id"]}'
        if row["case_id"] not in cases:
            raise ValueError("unknown review case")
        event_labels = review.get("system_event_labels", {}).get(key, {})
        finding_labels = review.get("system_finding_labels", {}).get(key, {})
        reasoning = review.get("reasoning_support_labels", {}).get(key, {})
        for event in _events(row):
            value = event_labels.get(event["id"], {})
            if value.get("signature") != _event_signature(row["case_id"], row["system"], event):
                raise ValueError(f"event signature mismatch: {key}/{event['id']}")
            if require_complete and value.get("label") not in EVENT_LABELS:
                raise ValueError(f"unreviewed event: {key}/{event['id']}")
        for finding in _findings(row):
            value = finding_labels.get(finding["id"], {})
            if value.get("signature") != _finding_signature(row["case_id"], row["system"], finding):
                raise ValueError(f"finding signature mismatch: {key}/{finding['id']}")
            if require_complete and value.get("label") not in FINDING_LABELS:
                raise ValueError(f"unreviewed finding: {key}/{finding['id']}")
            if require_complete and reasoning.get(finding["id"], {}).get("label") not in REASONING_LABELS:
                raise ValueError(f"unreviewed reasoning: {key}/{finding['id']}")
    return {"status": review["status"], "provenance": review["provenance"], "cases": len(cases)}
