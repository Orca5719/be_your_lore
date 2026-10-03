"""Benchmark 3.3A compact Judge variants on frozen retrieval inputs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from agent_pipeline_v2.batch_llm import call_json_batch


LABELS = {"consistent": "明确吻合", "contradiction": "明确矛盾", "uncertain": "不确定"}
RELATIONS = {"direct_support", "direct_conflict", "insufficient"}
NONACTUAL = {"speech", "belief", "plan", "dream", "inferred"}
VARIANTS = {"structured": 256, "short-reason": 320}
PROMPTS = {name: (Path(__file__).parent / "prompts" / f"judge_a_{name}.txt").read_text(encoding="utf-8")
           for name in VARIANTS}


def validate_answer(value: dict, evidence: list[dict], variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError("未知Judge实验版本")
    required = {"evidence_ids", "assessment", "verdict"}
    if variant == "short-reason":
        required.add("reason")
        if isinstance(value, dict) and "reason" not in value:
            raise ValueError("reason不能为空")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Judge输出字段无效")
    verdict = value["verdict"]
    if verdict not in LABELS:
        raise ValueError("verdict无效")
    ids = value["evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("evidence_ids无效或重复")
    aliases = {f"L{number}" for number in range(1, len(evidence) + 1)}
    if any(item not in aliases for item in ids):
        raise ValueError("evidence_id不存在")
    if verdict != "uncertain" and not ids:
        raise ValueError("明确结论必须提供证据ID")
    assessment = value["assessment"]
    fields = {"same_subject", "evidence_applicable", "relation", "can_both_be_true", "assumptions_required"}
    if not isinstance(assessment, dict) or set(assessment) != fields:
        raise ValueError("assessment字段无效")
    for field in ("same_subject", "evidence_applicable", "can_both_be_true"):
        if assessment[field] is not None and type(assessment[field]) is not bool:
            raise ValueError(f"assessment.{field}必须为布尔值或null")
    if assessment["relation"] not in RELATIONS:
        raise ValueError("assessment.relation无效")
    if type(assessment["assumptions_required"]) is not bool:
        raise ValueError("assessment.assumptions_required必须为布尔值")
    if variant == "short-reason" and (
        not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 80
    ):
        raise ValueError("reason须为不超过80字的非空文字")


def finalize_answer(value: dict, evidence: list[dict], variant: str) -> dict:
    validate_answer(value, evidence, variant)
    assessment = value["assessment"]
    verdict = value["verdict"]
    proven_common = (assessment["same_subject"] is True and assessment["evidence_applicable"] is True
                     and assessment["assumptions_required"] is False)
    conflict_proven = (proven_common and assessment["relation"] == "direct_conflict"
                       and assessment["can_both_be_true"] is False)
    support_proven = (proven_common and assessment["relation"] == "direct_support"
                      and assessment["can_both_be_true"] is True)
    result = {
        "verdict": verdict,
        "evidence_ids": list(value["evidence_ids"]),
        "citations": [{"evidence_id": alias, "chunk_id": evidence[int(alias[1:]) - 1]["id"]}
                      for alias in value["evidence_ids"]],
        "assessment": copy.deepcopy(assessment),
    }
    if variant == "short-reason":
        result["reason"] = value["reason"]
    if verdict == "contradiction" and not conflict_proven:
        result.update(verdict="uncertain", model_verdict=verdict, guard_reason="coexistence_not_ruled_out")
    elif verdict == "consistent" and not support_proven:
        result.update(verdict="uncertain", model_verdict=verdict, guard_reason="direct_support_not_proven")
    return result


def _messages(retrieval: dict, source: dict, variant: str) -> list[dict]:
    lore = [{"evidence_id": f"L{number}", "text": chunk["text"],
             "heading_path": chunk.get("heading_path", [])}
            for number, chunk in enumerate(source["evidence"], 1)]
    payload = {
        "fact": source.get("fact") or {"actors": source["event"].get("actors", []),
                                         "normalized_fact": source["event"].get("event")},
        "story_context": retrieval.get("extraction", {}).get("text", ""),
        "lore": lore,
    }
    return [{"role": "system", "content": PROMPTS[variant]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def _uncertain(source: dict, origin: str, status: str = "ok") -> dict:
    return {"event_id": source["event_id"], "event": source["event"],
            "evidence": source.get("evidence", []), "status": status,
            "verdict": "uncertain", "label": LABELS["uncertain"],
            "origin": origin, "evidence_ids": [], "citations": []}


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
            progress({"purpose": f"judge_a_{variant}", "window_id": len(reports) + 1,
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
    return {"schema_version": "agent-pipeline-v3.3-judge-a-v1", "stage": "judge", "variant": variant,
            "status": status, "events": copy.deepcopy(retrieval.get("events", [])), "items": items,
            "retrieval": copy.deepcopy(retrieval), "batch_size": batch_size,
            "batch_reports": reports, "request_seconds": time.perf_counter() - started,
            "failure_reasons": {"upstream_status": upstream, "failed_event_ids": failed}}
