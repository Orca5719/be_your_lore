"""Finding-level end-to-end conflict detection metrics."""

from __future__ import annotations

from typing import Any


VERDICT_CODES = {"一致": "consistent", "矛盾": "contradiction", "不确定": "uncertain"}


def _rate(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "total": total, "value": count / total if total else None}


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _mapped_ids(values: Any, key: str) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        identifier = value if isinstance(value, str) else value.get(key) if isinstance(value, dict) else None
        if isinstance(identifier, str):
            result.append(identifier)
    return result


def score_detection(dataset_cases: list[dict], run_rows: list[dict], review: dict) -> dict:
    """Score reviewed conflict findings and story-level false alarms."""
    rows = {row.get("case_id"): row for row in run_rows}
    tp = fp = fn = 0
    duplicate_conflicts = conflict_outputs = 0
    unsupported_claims = all_claims = 0
    omitted = omission_total = 0
    exact_stories = false_alarm_stories = 0
    per_case = {}

    for case in dataset_cases:
        case_id = case["id"]
        result = rows.get(case_id, {}).get("result", {})
        system_findings = {
            finding["id"]: finding
            for finding in result.get("findings", [])
            if isinstance(finding, dict) and isinstance(finding.get("id"), str)
        }
        judge_items = {
            item["event_id"]: item
            for item in result.get("judge", {}).get("items", [])
            if isinstance(item, dict) and isinstance(item.get("event_id"), str)
        }
        gold_findings = {finding["id"]: finding for finding in case.get("gold_findings", [])}
        gold_facts = {fact["id"]: fact for fact in case.get("gold_facts", [])}
        finding_labels = review.get("system_finding_labels", {}).get(case_id, {})
        finding_mapping = review.get("finding_mapping", {}).get(case_id, {})
        event_labels = review.get("system_event_labels", {}).get(case_id, {})
        event_mapping = review.get("event_mapping", {}).get(case_id, {})

        matched_gold = set()
        for gold_id in gold_findings:
            for finding_id in _mapped_ids(finding_mapping.get(gold_id, []), "finding_id"):
                if finding_id in system_findings and finding_labels.get(finding_id, {}).get("label") == "true_positive":
                    matched_gold.add(gold_id)
                    break
        case_tp = len(matched_gold)
        case_fn = len(gold_findings) - case_tp
        case_fp = 0
        case_duplicates = 0
        for finding_id, finding in system_findings.items():
            label = finding_labels.get(finding_id, {}).get("label")
            all_claims += 1
            unsupported_claims += label == "unsupported"
            if finding.get("verdict") != "contradiction":
                continue
            conflict_outputs += 1
            if label == "duplicate":
                duplicate_conflicts += 1
                case_duplicates += 1
                continue
            linked_gold = set(finding_labels.get(finding_id, {}).get("gold_finding_ids", [])) & set(gold_findings)
            if label in {"false_positive", "unsupported"} or (label == "true_positive" and not linked_gold):
                case_fp += 1

        reported_gold_facts = {
            fact_id
            for finding_id in matched_gold
            for fact_id in gold_findings[finding_id].get("gold_fact_ids", [])
        }
        case_omitted = case_omission_total = 0
        for fact_id, fact in gold_facts.items():
            if not fact.get("reportable"):
                continue
            expected = VERDICT_CODES.get(fact.get("expected_verdict"))
            correctly_judged = False
            for event_id in _mapped_ids(event_mapping.get(fact_id, []), "event_id"):
                if event_labels.get(event_id, {}).get("label") != "valid_checkable":
                    continue
                item = judge_items.get(event_id, {})
                if item.get("status") == "ok" and item.get("verdict") == expected:
                    correctly_judged = True
                    break
            if correctly_judged:
                case_omission_total += 1
                case_omitted += fact_id not in reported_gold_facts

        tp += case_tp
        fp += case_fp
        fn += case_fn
        omitted += case_omitted
        omission_total += case_omission_total
        exact = case_tp == len(gold_findings) and case_fp == 0
        exact_stories += exact
        false_alarm_stories += case_fp > 0
        per_case[case_id] = {
            "tp": case_tp,
            "fp": case_fp,
            "fn": case_fn,
            "duplicate_findings": case_duplicates,
            "report_omissions": case_omitted,
            "reportable_correct_judgements": case_omission_total,
            "exact_match": exact,
            "any_false_alarm": case_fp > 0,
        }

    story_total = len(dataset_cases)
    return {
        "schema_version": "agent-pipeline-v2-detection-metrics-v1",
        "conflict": _prf(tp, fp, fn),
        "duplicate_finding_rate": _rate(duplicate_conflicts, conflict_outputs),
        "report_omission_rate": _rate(omitted, omission_total),
        "unsupported_report_claim_rate": _rate(unsupported_claims, all_claims),
        "story_exact_match": _rate(exact_stories, story_total),
        "story_any_false_alarm_rate": _rate(false_alarm_stories, story_total),
        "per_case": per_case,
    }
