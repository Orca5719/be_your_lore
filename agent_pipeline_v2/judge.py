"""Evidence-bound v2 Judge with true micro-batched generation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from .batch_llm import call_json_batch
from .retrieval import validate_retrieval


SCHEMA_VERSION = "agent-pipeline-v2-judge-v1"
LABELS = {"consistent": "明确吻合", "contradiction": "明确矛盾", "uncertain": "不确定"}
NONACTUAL = {"speech", "belief", "plan", "dream", "inferred"}


def validate_answer(value: dict, evidence: list[dict]) -> None:
    if not isinstance(value, dict) or set(value) != {"verdict", "citations", "reason", "assessment"}:
        raise ValueError("判断字段必须为verdict/citations/reason/assessment")
    verdict = value["verdict"]
    if verdict not in LABELS:
        raise ValueError("verdict无效")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason不能为空")
    assessment = value["assessment"]
    if not isinstance(assessment, dict) or set(assessment) != {"same_subject", "evidence_applicable", "relation", "assumptions"}:
        raise ValueError("assessment字段无效")
    if any(assessment[key] is not None and type(assessment[key]) is not bool for key in ("same_subject", "evidence_applicable")):
        raise ValueError("assessment布尔字段无效")
    if assessment["relation"] not in {"direct_support", "direct_conflict", "insufficient"}:
        raise ValueError("relation无效")
    if not isinstance(assessment["assumptions"], list) or any(not isinstance(item, str) or not item.strip() for item in assessment["assumptions"]):
        raise ValueError("assumptions须为文字数组")
    citations = value["citations"]
    if not isinstance(citations, list) or len(citations) > len(evidence):
        raise ValueError("citations无效")
    if verdict != "uncertain" and not citations:
        raise ValueError("明确结论必须引用证据")
    aliases = {f"L{number}": chunk for number, chunk in enumerate(evidence, 1)}
    seen = set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"evidence_id", "quote"}:
            raise ValueError("citation字段无效")
        alias = citation["evidence_id"]
        quote = citation["quote"]
        if alias not in aliases or alias in seen:
            raise ValueError("citation编号不存在或重复")
        if not isinstance(quote, str) or not quote.strip() or quote not in aliases[alias]["text"]:
            raise ValueError("citation原文不属于对应设定")
        seen.add(alias)


def _messages(prompt: str, retrieval_report: dict, source: dict) -> list[dict]:
    event = source["event"]
    lore = [
        {"evidence_id": f"L{number}", "text": chunk["text"], "heading_path": chunk.get("heading_path", [])}
        for number, chunk in enumerate(source["evidence"], 1)
    ]
    payload = {
        "event": {key: event[key] for key in ("actors", "event", "mental_state", "explicit", "modality", "conditions", "check_reason")},
        "story_context": retrieval_report["extraction"]["text"],
        "lore": lore,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _metric_totals(batch_reports: list[dict]) -> dict:
    calls = [call for report in batch_reports for call in report["batch_calls"]]
    timings = [call.get("timing", {}) for call in calls]
    return {
        "main_batch_count": sum(call.get("attempt") == 1 for call in calls),
        "retry_batch_count": sum(call.get("attempt") == 2 for call in calls),
        "retry_fact_count": sum(report.get("retry_count", 0) for report in batch_reports),
        "generation_seconds": sum(float(timing.get("seconds", 0) or 0) for timing in timings),
        "useful_input_tokens": sum(int(timing.get("useful_input_tokens", sum(timing.get("input_tokens", []))) or 0) for timing in timings),
        "padded_input_tokens": sum(int(timing.get("padded_input_tokens", 0) or 0) for timing in timings),
        "generated_tokens": sum(int(timing.get("total_generated_tokens", sum(timing.get("generated_tokens", []))) or 0) for timing in timings),
    }


def validate_judge(report: dict) -> None:
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v2 Judge schema_version无效")
    if report.get("stage") != "judge" or report.get("status") not in {"ok", "partial", "error"}:
        raise ValueError("v2 Judge阶段或状态无效")
    retrieval = report.get("retrieval")
    validate_retrieval(retrieval)
    items = report.get("items")
    if not isinstance(items, list) or len(items) != len(retrieval["items"]):
        raise ValueError("Judge items数量无效")
    expected = {item["event_id"] for item in retrieval["items"]}
    actual = [item.get("event_id") for item in items]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("Judge事件ID缺失或重复")
    for item in items:
        if item.get("status") not in {"ok", "error"} or item.get("verdict") not in LABELS:
            raise ValueError("Judge item状态或结论无效")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("batch_size_requested") != report.get("batch_size"):
        raise ValueError("Judge metrics缺失或batch不匹配")


def judge_events(retrieval_report: dict, batch_size: int = 1, device: str = "auto", llm=None, progress=None) -> dict:
    validate_retrieval(retrieval_report)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size必须为正整数")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device必须为auto/cpu/cuda")
    started = time.perf_counter()
    upstream = copy.deepcopy(retrieval_report)
    loaded_here = llm is None
    if llm is None:
        from .model import V2QwenJudge

        llm = V2QwenJudge(device)
    prompt = (Path(__file__).parent / "prompts" / "judge_v1.txt").read_text(encoding="utf-8")
    items_by_id = {}
    pending = []
    for source in upstream["items"]:
        event_id = source["event_id"]
        base = {"event_id": event_id, "event": source["event"], "evidence": source["evidence"], "citations": []}
        if source["status"] == "error":
            base.update(status="error", verdict="uncertain", label=LABELS["uncertain"], origin="retrieval_failure", reason="检索失败，无法完成判断", error=source.get("error", "检索失败"))
            items_by_id[event_id] = base
        elif source["event"]["modality"] in NONACTUAL:
            base.update(status="ok", verdict="uncertain", label=LABELS["uncertain"], origin="program_nonactual_scope", reason="非客观叙述不作为已经发生的事实核对")
            items_by_id[event_id] = base
        elif not source["evidence"]:
            base.update(status="ok", verdict="uncertain", label=LABELS["uncertain"], origin="program_no_evidence", reason="没有候选设定，无法明确核对")
            items_by_id[event_id] = base
        else:
            pending.append(source)
    batch_reports = []
    for offset in range(0, len(pending), batch_size):
        sources = pending[offset : offset + batch_size]
        if progress:
            progress({"window_id": len(batch_reports) + 1, "purpose": "batched_judge", "target_ids": [source["event_id"] for source in sources], "batch_size": len(sources)})
        requests = [{"request_id": source["event_id"], "messages": _messages(prompt, upstream, source)} for source in sources]
        validators = [(lambda evidence: lambda value: validate_answer(value, evidence))(source["evidence"]) for source in sources]
        values, batch_report = call_json_batch(llm, requests, max_output=768, validators=validators)
        batch_reports.append(batch_report)
        rows = {row["request_id"]: row for row in batch_report["rows"]}
        for source, value in zip(sources, values):
            event_id = source["event_id"]
            base = {"event_id": event_id, "event": source["event"], "evidence": source["evidence"], "citations": [], "call": rows[event_id]}
            if value is None:
                base.update(status="error", verdict="uncertain", label=LABELS["uncertain"], origin="judge_failure", reason="模型未返回有效判断", error=rows[event_id]["error"])
            else:
                citations = [
                    {**citation, "chunk_id": source["evidence"][int(citation["evidence_id"][1:]) - 1]["id"]}
                    for citation in value["citations"]
                ]
                verdict = value["verdict"]
                assessment = value["assessment"]
                expected_relation = "direct_support" if verdict == "consistent" else "direct_conflict"
                if verdict != "uncertain" and (
                    assessment["same_subject"] is not True
                    or assessment["evidence_applicable"] is not True
                    or assessment["assumptions"]
                    or assessment["relation"] != expected_relation
                ):
                    base.update(status="ok", verdict="uncertain", label=LABELS["uncertain"], origin="program_evidence_scope_guard", reason="证据对象或适用范围不足以支持明确结论", model_verdict=verdict, model_reason=value["reason"], assessment=assessment, citations=citations)
                else:
                    base.update(status="ok", verdict=verdict, label=LABELS[verdict], origin="model", reason=value["reason"], assessment=assessment, citations=citations)
            items_by_id[event_id] = base
    items = [items_by_id[source["event_id"]] for source in upstream["items"]]
    failures = [item["event_id"] for item in items if item["status"] == "error"]
    status = "error" if upstream["status"] == "error" or (items and len(failures) == len(items)) else "partial" if failures or upstream["status"] != "ok" else "ok"
    metrics = {"batch_size_requested": batch_size, **_metric_totals(batch_reports)}
    metrics["padding_ratio"] = metrics["padded_input_tokens"] / (metrics["useful_input_tokens"] + metrics["padded_input_tokens"]) if metrics["useful_input_tokens"] + metrics["padded_input_tokens"] else 0.0
    result = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": "judge-v1",
        "stage": "judge",
        "status": status,
        "batch_size": batch_size,
        "events": upstream["events"],
        "items": items,
        "retrieval": upstream,
        "metrics": metrics,
        "batch_reports": batch_reports,
        "failure_reasons": {"upstream_status": upstream["status"], "failed_event_ids": failures},
        "processing_complete": status == "ok",
        "model_loaded_this_request": loaded_here,
        "device": getattr(llm, "device", device),
        "model_load_seconds": getattr(llm, "load_seconds", None),
        "request_seconds": time.perf_counter() - started,
    }
    validate_judge(result)
    return result
