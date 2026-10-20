"""Second 3.3A ablation: original assessment semantics with bounded reasoning."""

from __future__ import annotations

import copy
from pathlib import Path
import time

from agent_pipeline_v2.batch_llm import call_json_batch
from agent_pipeline_v2_1.judge import apply_scope_guard

from .judge_a import LABELS, NONACTUAL, RELATIONS, _messages as baseline_messages, _uncertain


VARIANTS = {"assumptions-list": 320, "rationale-120": 448, "rationale-240": 640}
REASON_LIMITS = {"rationale-120": 120, "rationale-240": 240}
PROMPTS = {name: (Path(__file__).parent / "prompts" / f"judge_a2_{name}.txt").read_text(encoding="utf-8")
           for name in VARIANTS}


def validate_answer(value: dict, evidence: list[dict], variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError("未知Judge实验版本")
    fields = {"evidence_ids", "assessment", "verdict"}
    if variant in REASON_LIMITS:
        fields.add("reason")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Judge输出字段无效")
    if value["verdict"] not in LABELS:
        raise ValueError("verdict无效")
    ids = value["evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("evidence_ids无效或重复")
    aliases = {f"L{number}" for number in range(1, len(evidence) + 1)}
    if any(item not in aliases for item in ids):
        raise ValueError("evidence_id不存在")
    if value["verdict"] != "uncertain" and not ids:
        raise ValueError("明确结论必须提供证据ID")
    assessment = value["assessment"]
    assessment_fields = {"same_subject", "evidence_applicable", "relation", "can_both_be_true", "assumptions"}
    if not isinstance(assessment, dict) or set(assessment) != assessment_fields:
        raise ValueError("assessment字段无效")
    for field in ("same_subject", "evidence_applicable", "can_both_be_true"):
        if assessment[field] is not None and type(assessment[field]) is not bool:
            raise ValueError(f"assessment.{field}必须为布尔值或null")
    if assessment["relation"] not in RELATIONS:
        raise ValueError("assessment.relation无效")
    assumptions = assessment["assumptions"]
    if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip() for item in assumptions):
        raise ValueError("assessment.assumptions须为文字数组")
    if variant in REASON_LIMITS and (
        not isinstance(value["reason"], str) or not value["reason"].strip()
        or len(value["reason"]) > REASON_LIMITS[variant]
    ):
        raise ValueError("reason长度或内容无效")


def finalize_answer(value: dict, evidence: list[dict], variant: str) -> dict:
    validate_answer(value, evidence, variant)
    guarded = apply_scope_guard(value)
    result = {
        "verdict": guarded["verdict"],
        "evidence_ids": list(value["evidence_ids"]),
        "citations": [{"evidence_id": alias, "chunk_id": evidence[int(alias[1:]) - 1]["id"]}
                      for alias in value["evidence_ids"]],
        "assessment": copy.deepcopy(value["assessment"]),
    }
    if variant in REASON_LIMITS:
        result["reason"] = value["reason"]
    for field in ("model_verdict", "guard_reason"):
        if field in guarded:
            result[field] = guarded[field]
    return result


def _messages(retrieval: dict, source: dict, variant: str) -> list[dict]:
    messages = baseline_messages(retrieval, source, "structured")
    messages[0] = {"role": "system", "content": PROMPTS[variant]}
    return messages


def judge_frozen_retrieval(retrieval: dict, llm, variant: str, batch_size: int = 8, progress=None) -> dict:
    if variant not in VARIANTS:
        raise ValueError("未知Judge实验版本")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size必须为正整数")
    started = time.perf_counter()
    items_by_id: dict[str, dict] = {}
    pending = []
    for source in retrieval.get("items", []):
        event = source.get("event", {})
        if source.get("status") != "ok":
            items_by_id[source["event_id"]] = _uncertain(source, "retrieval_failure", "error")
        elif event.get("modality") in NONACTUAL:
            items_by_id[source["event_id"]] = _uncertain(source, "program_nonactual_scope")
        elif not source.get("evidence"):
            items_by_id[source["event_id"]] = _uncertain(source, "program_no_evidence")
        else:
            pending.append(source)
    reports = []
    for offset in range(0, len(pending), batch_size):
        group = pending[offset:offset + batch_size]
        if progress:
            progress({"purpose": f"judge_a2_{variant}", "window_id": len(reports) + 1,
                      "target_ids": [source["event_id"] for source in group]})
        requests = [{"request_id": source["event_id"], "messages": _messages(retrieval, source, variant)}
                    for source in group]
        validators = [(lambda evidence: lambda value: validate_answer(value, evidence, variant))(source["evidence"])
                      for source in group]
        values, report = call_json_batch(llm, requests, max_output=VARIANTS[variant], validators=validators)
        reports.append(report)
        transport = {row["request_id"]: row for row in report["rows"]}
        for source, value in zip(group, values):
            event_id = source["event_id"]
            if value is None:
                item = _uncertain(source, "judge_failure", "error")
                item["error"] = transport[event_id].get("error")
            else:
                final = finalize_answer(value, source["evidence"], variant)
                item = {"event_id": event_id, "event": source["event"], "evidence": source["evidence"],
                        "status": "ok", "verdict": final["verdict"], "label": LABELS[final["verdict"]],
                        "origin": "program_scope_guard" if "guard_reason" in final else "model", **final}
            items_by_id[event_id] = item
    items = [items_by_id[source["event_id"]] for source in retrieval.get("items", [])]
    failed = [item["event_id"] for item in items if item["status"] == "error"]
    upstream = retrieval.get("status", "error")
    status = "error" if upstream == "error" or (items and len(failed) == len(items)) else "partial" if failed or upstream != "ok" else "ok"
    return {"schema_version": "agent-pipeline-v3.3-judge-a2-v1", "stage": "judge", "variant": variant,
            "status": status, "events": copy.deepcopy(retrieval.get("events", [])), "items": items,
            "retrieval": copy.deepcopy(retrieval), "batch_size": batch_size,
            "batch_reports": reports, "request_seconds": time.perf_counter() - started,
            "failure_reasons": {"upstream_status": upstream, "failed_event_ids": failed}}
