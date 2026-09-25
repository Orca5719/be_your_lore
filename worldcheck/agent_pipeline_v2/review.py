"""Human-review ledger generation, proposals, and strict validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


EVENT_LABELS = {"valid_checkable", "overselected", "hallucinated", "duplicate"}
FINDING_LABELS = {"true_positive", "false_positive", "duplicate", "unsupported"}


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _case_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in result:
            raise ValueError(f"运行结果case_id无效或重复：{case_id}")
        result[case_id] = row
    return result


def _events(result: dict) -> list[dict]:
    judge = result.get("judge", {}) if isinstance(result, dict) else {}
    rows = judge.get("events", [])
    return rows if isinstance(rows, list) else []


def _findings(result: dict) -> list[dict]:
    rows = result.get("findings", []) if isinstance(result, dict) else []
    return rows if isinstance(rows, list) else []


def _ids(rows: list[dict], label: str) -> set[str]:
    result = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"{label}ID无效或重复：{identifier}")
        result.add(identifier)
    return result


def build_review_template(dataset: dict, run_rows: list[dict]) -> dict:
    """Create an exhaustive pending ledger without making semantic decisions."""
    cases = {case["id"]: case for case in dataset.get("cases", [])}
    rows = _case_rows(run_rows)
    unknown = set(rows) - set(cases)
    if unknown:
        raise ValueError(f"运行结果包含未知案例：{sorted(unknown)}")
    event_mapping, finding_mapping = {}, {}
    system_event_labels, system_finding_labels = {}, {}
    for case_id, case in cases.items():
        result = rows.get(case_id, {}).get("result", {})
        events = _events(result)
        findings = _findings(result)
        _ids(events, f"{case_id}事件")
        _ids(findings, f"{case_id}finding")
        event_mapping[case_id] = {gold["id"]: [] for gold in case.get("gold_facts", [])}
        finding_mapping[case_id] = {gold["id"]: [] for gold in case.get("gold_findings", [])}
        system_event_labels[case_id] = {
            event["id"]: {"label": "pending", "gold_ids": [], "note": ""} for event in events
        }
        system_finding_labels[case_id] = {
            finding["id"]: {"label": "pending", "gold_finding_ids": [], "note": ""} for finding in findings
        }
    return {
        "schema_version": "agent-pipeline-v2-review-v1",
        "status": "pending_human_review",
        "dataset_sha256": _digest(dataset),
        "run_rows_sha256": _digest(run_rows),
        "event_mapping": event_mapping,
        "finding_mapping": finding_mapping,
        "system_event_labels": system_event_labels,
        "system_finding_labels": system_finding_labels,
        "retrieval_relevance_overrides": {},
        "reasoning_support_labels": {},
        "error_attribution": {},
        "instruction": "逐一审核所有事件和finding；proposal只能作为建议，不能代替人工确认。",
    }


def _event_text(event: dict) -> str:
    actors = "".join(event.get("actors", [])) if isinstance(event.get("actors"), list) else str(event.get("actors", ""))
    return actors + str(event.get("event", ""))


def _source_texts(result: dict) -> dict[str, str]:
    extraction = result.get("judge", {}).get("retrieval", {}).get("extraction", {})
    spans = extraction.get("spans", {}) if isinstance(extraction, dict) else {}
    return {str(key): str(value.get("text", "")) for key, value in spans.items() if isinstance(value, dict)}


def propose_review(dataset: dict, run_rows: list[dict]) -> dict:
    """Propose mappings using exact anchors first, then conservative text overlap."""
    template = build_review_template(dataset, run_rows)
    rows = _case_rows(run_rows)
    for case in dataset.get("cases", []):
        case_id = case["id"]
        result = rows.get(case_id, {}).get("result", {})
        events = {event["id"]: event for event in _events(result)}
        spans = _source_texts(result)
        for gold in case.get("gold_facts", []):
            candidates = []
            anchors = [anchor["text"] for anchor in gold.get("source_anchors", [])]
            gold_subject = str(gold.get("subject", ""))
            gold_text = str(gold.get("normalized_fact", ""))
            for event_id, event in events.items():
                score = 0
                source_ids = [str(value) for value in event.get("source_ids", [])]
                source_blob = "".join(spans.get(source_id, "") for source_id in source_ids)
                event_blob = _event_text(event)
                if any(anchor and anchor in source_blob for anchor in anchors):
                    score += 100
                if gold_subject and gold_subject in event_blob:
                    score += 20
                if gold_text and (gold_text in event_blob or gold_text in source_blob):
                    score += 50
                if score:
                    candidates.append({"event_id": event_id, "score": score, "reason": "anchor_or_subject_overlap"})
            candidates.sort(key=lambda row: (-row["score"], row["event_id"]))
            if candidates:
                template["event_mapping"][case_id][gold["id"]] = candidates[:3]
        for finding in _findings(result):
            event_ids = [str(value) for value in finding.get("event_ids", [])]
            for gold_finding in case.get("gold_findings", []):
                gold_events = template["event_mapping"][case_id]
                mapped = set()
                for gold_id in gold_finding.get("gold_fact_ids", []):
                    mapped.update(item["event_id"] for item in gold_events.get(gold_id, []) if isinstance(item, dict))
                if mapped.intersection(event_ids):
                    template["finding_mapping"][case_id][gold_finding["id"]].append({"finding_id": finding["id"], "score": 100, "reason": "mapped_event_overlap"})
    return template


def validate_review(dataset: dict, run_rows: list[dict], review: dict, *, require_complete: bool = True) -> dict:
    """Validate that every emitted object has an explicit, well-formed disposition."""
    if not isinstance(review, dict) or review.get("schema_version") != "agent-pipeline-v2-review-v1":
        raise ValueError("审核账本schema_version无效")
    if require_complete and review.get("status") != "reviewed":
        raise ValueError("审核账本尚未标记为reviewed")
    cases = {case["id"]: case for case in dataset.get("cases", [])}
    rows = _case_rows(run_rows)
    for case_id, case in cases.items():
        if case_id not in rows:
            raise ValueError(f"缺少案例运行结果：{case_id}")
        result = rows[case_id].get("result", {})
        events = {event["id"]: event for event in _events(result)}
        findings = {finding["id"]: finding for finding in _findings(result)}
        gold_ids = {gold["id"] for gold in case.get("gold_facts", [])}
        gold_finding_ids = {finding["id"] for finding in case.get("gold_findings", [])}
        labels = review.get("system_event_labels", {}).get(case_id, {})
        finding_labels = review.get("system_finding_labels", {}).get(case_id, {})
        mappings = review.get("event_mapping", {}).get(case_id, {})
        finding_mappings = review.get("finding_mapping", {}).get(case_id, {})
        for event_id, event in events.items():
            if event_id not in labels or labels[event_id].get("label") == "pending":
                if require_complete:
                    raise ValueError(f"未审核事件：{case_id}/{event_id}")
                continue
            label = labels[event_id]
            if label.get("label") not in EVENT_LABELS:
                raise ValueError(f"事件标签无效：{case_id}/{event_id}")
            linked = label.get("gold_ids", [])
            if any(value not in gold_ids for value in linked):
                raise ValueError(f"事件引用了不存在的gold fact：{case_id}/{event_id}")
            if label["label"] in {"hallucinated", "overselected"} and not str(label.get("note", "")).strip():
                raise ValueError(f"事件标签缺少说明：{case_id}/{event_id}")
            if label["label"] == "duplicate":
                origin = label.get("duplicate_of")
                if not isinstance(origin, str) or origin not in events or origin == event_id:
                    raise ValueError(f"duplicate_of无效：{case_id}/{event_id}")
        for finding_id in findings:
            if finding_id not in finding_labels or finding_labels[finding_id].get("label") == "pending":
                if require_complete:
                    raise ValueError(f"未审核finding：{case_id}/{finding_id}")
                continue
            label = finding_labels[finding_id]
            if label.get("label") not in FINDING_LABELS:
                raise ValueError(f"finding标签无效：{case_id}/{finding_id}")
            linked = label.get("gold_finding_ids", [])
            if any(value not in gold_finding_ids for value in linked):
                raise ValueError(f"finding引用了不存在的gold finding：{case_id}/{finding_id}")
            if label["label"] in {"false_positive", "unsupported"} and not str(label.get("note", "")).strip():
                raise ValueError(f"finding标签缺少说明：{case_id}/{finding_id}")
            if label["label"] == "duplicate":
                origin = label.get("duplicate_of")
                if not isinstance(origin, str) or origin not in findings or origin == finding_id:
                    raise ValueError(f"finding duplicate_of无效：{case_id}/{finding_id}")
        for gold_id, event_ids in mappings.items():
            if gold_id not in gold_ids or any((item if isinstance(item, str) else item.get("event_id")) not in events for item in event_ids):
                raise ValueError(f"事件映射引用无效：{case_id}/{gold_id}")
        for gold_id, finding_ids in finding_mappings.items():
            if gold_id not in gold_finding_ids or any((item if isinstance(item, str) else item.get("finding_id")) not in findings for item in finding_ids):
                raise ValueError(f"finding映射引用无效：{case_id}/{gold_id}")
    return {"status": review.get("status"), "cases": len(cases), "reviewed_events": sum(len(_events(row.get("result", {}))) for row in rows.values()), "reviewed_findings": sum(len(_findings(row.get("result", {}))) for row in rows.values())}
