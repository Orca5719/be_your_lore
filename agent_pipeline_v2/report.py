"""Deterministic author-facing report for Agent Pipeline v2."""

from __future__ import annotations

import copy

from .judge import validate_judge


def build_report(judge_report: dict) -> dict:
    validate_judge(judge_report)
    judge = copy.deepcopy(judge_report)
    counts = {"consistent": 0, "contradiction": 0, "uncertain": 0, "failed": 0}
    findings = []
    for number, item in enumerate(judge["items"], 1):
        if item["status"] == "error":
            counts["failed"] += 1
        else:
            counts[item["verdict"]] += 1
        findings.append({
            "id": f"R{number}",
            "event_id": item["event_id"],
            "actors": item["event"]["actors"],
            "event": item["event"]["event"],
            "verdict": item["verdict"],
            "label": item["label"],
            "reason": item["reason"],
            "citations": item["citations"],
            "status": item["status"],
            "origin": item["origin"],
        })
    if counts["contradiction"]:
        verdict = "contradiction"
        label = "发现与候选设定明确矛盾的事件"
    elif counts["failed"] or counts["uncertain"]:
        verdict = "uncertain"
        label = "存在未能明确核对的事件"
    else:
        verdict = "consistent"
        label = "已核对事件均与候选设定明确吻合"
    extraction = judge["retrieval"]["extraction"]
    return {
        "schema_version": "agent-pipeline-v2-report-v1",
        "stage": "report",
        "status": judge["status"],
        "processing_complete": judge["status"] == "ok",
        "summary": {
            "verdict": verdict,
            "label": label,
            "counts": counts,
            "checked_event_count": len(judge["items"]),
            "ignored_span_count": len(extraction["ignored_spans"]),
            "non_event_span_count": len(extraction["non_event_span_ids"]),
        },
        "findings": findings,
        "ignored_spans": extraction["ignored_spans"],
        "non_event_span_ids": extraction["non_event_span_ids"],
        "uncovered_span_ids": extraction.get("uncovered_span_ids", []),
        "judge": judge,
        "notice": "仅核对一步提取器选出的事件；不确定不是矛盾，未覆盖片段不代表无矛盾。相似度不是事实正确率。",
    }

