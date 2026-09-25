"""Earliest-stage error attribution for reviewed story benchmark outputs."""

from __future__ import annotations


CODE_TO_LABEL = {"consistent": "一致", "contradiction": "矛盾", "uncertain": "不确定"}


def _ids(values):
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        identifier = value if isinstance(value, str) else value.get("event_id") if isinstance(value, dict) else None
        if isinstance(identifier, str):
            result.append(identifier)
    return result


def _error(case_id, error_type, stage, explanation, **links):
    return {
        "id": "", "case_id": case_id, "error_type": error_type, "primary_stage": stage,
        "contributing_stages": [], "automatic_explanation": explanation,
        "review_status": "pending", "reviewer_note": "", **links,
    }


def _apply_reviewed_overrides(errors, review):
    overrides = review.get("error_attribution", {})
    used = set()
    for case_id, values in overrides.items():
        if not isinstance(values, dict):
            continue
        candidates = [error for error in errors if error["case_id"] == case_id]
        for override in values.values():
            if not isinstance(override, dict):
                continue
            target = None
            for error in candidates:
                if error["id"] in used:
                    continue
                if override.get("event_id") and override.get("event_id") in error.get("event_ids", []):
                    target = error
                    break
                if override.get("gold_id") and override.get("gold_id") in error.get("gold_ids", []):
                    target = error
                    break
                if override.get("finding_id") and override.get("finding_id") in error.get("finding_ids", []):
                    target = error
                    break
            if target is None:
                target = next((error for error in candidates if error["id"] not in used), None)
            if target is None:
                continue
            used.add(target["id"])
            for key in ("error_type", "primary_stage", "contributing_stages", "review_status", "reviewer_note"):
                if key in override:
                    target[key] = override[key]


def build_error_attribution(dataset: dict, run_rows: list[dict], review: dict, quality: dict) -> dict:
    errors = []
    rows = {row.get("case_id"): row for row in run_rows}
    for case in dataset.get("cases", []):
        case_id = case["id"]
        result = rows.get(case_id, {}).get("result", {})
        judge = result.get("judge", {})
        events = {event["id"]: event for event in judge.get("events", []) if isinstance(event, dict) and isinstance(event.get("id"), str)}
        items = {item["event_id"]: item for item in judge.get("items", []) if isinstance(item, dict) and isinstance(item.get("event_id"), str)}
        retrieval = {item["event_id"]: item for item in judge.get("retrieval", {}).get("items", []) if isinstance(item, dict) and isinstance(item.get("event_id"), str)}
        event_labels = review.get("system_event_labels", {}).get(case_id, {})
        mapping = review.get("event_mapping", {}).get(case_id, {})
        for event_id in sorted(events):
            label = event_labels.get(event_id, {}).get("label")
            if label == "hallucinated":
                error = _error(case_id, "EXTRACTION_HALLUCINATION", "extractor", "事件命题没有被原文和必要上下文支持。", event_ids=[event_id])
                if event_id in items:
                    error["contributing_stages"].append("judge")
                errors.append(error)
            elif label == "overselected":
                errors.append(_error(case_id, "EXTRACTION_OVERSELECT", "extractor", "原文存在该片段，但审核认为它是普通动作或流程细节。", event_ids=[event_id]))

        valid_event_labels = {event_id for event_id, label in event_labels.items() if label.get("label") == "valid_checkable"}
        for fact in case.get("gold_facts", []):
            gold_id = fact["id"]
            mapped = [event_id for event_id in _ids(mapping.get(gold_id, [])) if event_id in valid_event_labels and event_id in events]
            if not mapped:
                errors.append(_error(case_id, "EXTRACTION_MISS", "extractor", "gold fact没有对应的有效提取事件。", gold_ids=[gold_id]))
                continue
            expected = fact.get("expected_verdict")
            minimum_sets = [set(group) for group in fact.get("minimum_evidence_sets", []) if group]
            retrieval_ok = False
            for event_id in mapped:
                evidence = {chunk.get("id") for chunk in retrieval.get(event_id, {}).get("evidence", []) if isinstance(chunk, dict)}
                retrieval_ok = retrieval_ok or not minimum_sets or any(group.issubset(evidence) for group in minimum_sets)
            if minimum_sets and not retrieval_ok:
                error = _error(case_id, "RETRIEVAL_MISS", "retrieval", "有效事件已提取，但Top-K没有包含充分证据组合。", gold_ids=[gold_id], event_ids=mapped)
                if any(event_id in items for event_id in mapped):
                    error["contributing_stages"].append("judge")
                errors.append(error)
                continue
            for event_id in mapped:
                item = items.get(event_id, {})
                predicted = CODE_TO_LABEL.get(item.get("verdict"))
                if item.get("status") == "ok" and predicted and predicted != expected:
                    error_type = "JUDGE_FALSE_POSITIVE" if predicted == "矛盾" else "JUDGE_FALSE_NEGATIVE"
                    errors.append(_error(case_id, error_type, "judge", "Judge已收到有效事件，但最终分类与gold verdict不一致。", gold_ids=[gold_id], event_ids=[event_id], judge_item_ids=[event_id]))
                    break
                support = review.get("reasoning_support_labels", {}).get(case_id, {}).get(event_id, {})
                if support.get("label") == "unsupported":
                    errors.append(_error(case_id, "UNSUPPORTED_REASONING", "judge", "审核认为Judge理由没有被证据支持。", gold_ids=[gold_id], event_ids=[event_id], judge_item_ids=[event_id]))

        finding_labels = review.get("system_finding_labels", {}).get(case_id, {})
        findings = result.get("findings", [])
        for finding in findings:
            finding_id = finding.get("id") if isinstance(finding, dict) else None
            if finding_id and finding_labels.get(finding_id, {}).get("label") == "duplicate":
                errors.append(_error(case_id, "DUPLICATE_FINDING", "report", "报告重复输出了同一冲突finding。", finding_ids=[finding_id]))
        report_omissions = quality.get("per_case", {}).get(case_id, {}).get("report_omissions", 0)
        for index in range(report_omissions):
            errors.append(_error(case_id, "REPORT_OMISSION", "report", "正确判断的可报告事实没有出现在最终报告。", gold_ids=[f"omitted:{index + 1}"]))

    for index, error in enumerate(errors, 1):
        error["id"] = f"AT-{index:04d}"
    _apply_reviewed_overrides(errors, review)
    return {
        "schema_version": "agent-pipeline-v2-error-attribution-v1",
        "status": "reviewed" if errors and all(error["review_status"] == "reviewed" for error in errors) else "pending_review",
        "errors": errors,
        "counts": {error_type: sum(error["error_type"] == error_type for error in errors) for error_type in sorted({error["error_type"] for error in errors})},
    }
