"""Agent Pipeline 2.1 story orchestration built on frozen v2 extraction."""

from __future__ import annotations

import copy

from agent_pipeline_v2.batch_llm import call_json_batch
from agent_pipeline_v2.extractor import extract_events, validate_extraction
from agent_pipeline_v2.retrieval import event_query

from .judge import validate_model_answer
from .oracle_benchmark import build_messages, finalize_answer


CHECK_DIMENSIONS = {
    "mechanism": ["physical_rule", "world_rule"],
    "knowledge_relation": ["character_knowledge", "character_relation", "identity"],
    "state_time_space": ["time", "space", "physical_rule"],
    "consequence_support": ["causality", "world_rule"],
}
NONACTUAL = {"speech", "belief", "plan", "dream", "inferred"}
LABELS = {"consistent": "明确吻合", "contradiction": "明确矛盾", "uncertain": "不确定"}


def event_fact(event: dict) -> dict:
    dimensions = CHECK_DIMENSIONS[event["check_reason"]]
    return {
        "subject": event["actors"][0],
        "actors": list(event["actors"]),
        "normalized_fact": event["event"],
        "dimension": dimensions[0],
        "dimensions": dimensions,
        "check_reason": event["check_reason"],
    }


def retrieve_story_events(extraction: dict, retriever, method: str, metadata_filter: bool, k: int = 5, progress=None) -> dict:
    validate_extraction(extraction)
    items = []
    for number, event in enumerate(extraction["events"], 1):
        if progress:
            progress({"purpose": "v2.1_retrieval", "window_id": number, "target_ids": [event["id"]]})
        fact = event_fact(event)
        query = event_query(event, extraction["spans"])
        base = {"event_id": event["id"], "event": event, "fact": fact, "query": query, "evidence": []}
        try:
            result = retriever.search_fact(fact, query, method=method, metadata_filter=metadata_filter, k=k)
            base.update(status="ok", evidence=result["results"], retrieval=result)
        except (ValueError, OSError, RuntimeError) as exc:
            base.update(status="error", error=str(exc))
        items.append(base)
    failures = [item["event_id"] for item in items if item["status"] == "error"]
    status = "error" if extraction["status"] == "error" or (items and len(failures) == len(items)) else "partial" if failures or extraction["status"] != "ok" else "ok"
    return {
        "schema_version": "agent-pipeline-v2.1-story-retrieval-v1",
        "stage": "retrieval",
        "status": status,
        "method": method,
        "metadata_filter": metadata_filter,
        "top_k": k,
        "extraction": copy.deepcopy(extraction),
        "events": copy.deepcopy(extraction["events"]),
        "items": items,
        "failure_reasons": {"upstream_status": extraction["status"], "failed_event_ids": failures},
    }


def _uncertain_item(source: dict, origin: str, reason: str, status: str = "ok") -> dict:
    return {
        "event_id": source["event_id"], "event": source["event"], "evidence": source["evidence"],
        "status": status, "verdict": "uncertain", "label": LABELS["uncertain"],
        "origin": origin, "reason": reason, "citations": [],
    }


def judge_story_events(retrieval: dict, llm, batch_size: int = 8, progress=None) -> dict:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size必须为正整数")
    items_by_id = {}
    pending = []
    for source in retrieval["items"]:
        event = source["event"]
        if source["status"] == "error":
            items_by_id[source["event_id"]] = _uncertain_item(source, "retrieval_failure", "检索失败，无法判断", "error")
        elif event["modality"] in NONACTUAL:
            items_by_id[source["event_id"]] = _uncertain_item(source, "program_nonactual_scope", "非客观叙述不作为已发生事实核对")
        elif not source["evidence"]:
            items_by_id[source["event_id"]] = _uncertain_item(source, "program_no_evidence", "没有候选设定，无法判断")
        else:
            pending.append(source)
    batch_reports = []
    for offset in range(0, len(pending), batch_size):
        group = pending[offset : offset + batch_size]
        if progress:
            progress({"purpose": "v2.1_judge", "window_id": len(batch_reports) + 1, "target_ids": [row["event_id"] for row in group]})
        requests = []
        validators = []
        for source in group:
            wire = {"fixture_id": source["event_id"], "fact": source["fact"], "story_context": retrieval["extraction"]["text"], "lore": source["evidence"]}
            requests.append({"request_id": source["event_id"], "messages": build_messages(wire, "v2.1")})
            validators.append((lambda evidence: lambda value: validate_model_answer(value, evidence))(source["evidence"]))
        values, batch_report = call_json_batch(llm, requests, max_output=768, validators=validators)
        batch_reports.append(batch_report)
        transports = {row["request_id"]: row for row in batch_report["rows"]}
        for source, value in zip(group, values):
            event_id = source["event_id"]
            if value is None:
                item = _uncertain_item(source, "judge_failure", "模型未返回有效判断", "error")
                item["error"] = transports[event_id].get("error")
            else:
                final = finalize_answer(value, source["evidence"], "v2.1")
                citations = [
                    {**citation, "chunk_id": source["evidence"][int(citation["evidence_id"][1:]) - 1]["id"]}
                    for citation in final["citations"]
                ]
                item = {
                    "event_id": event_id, "event": source["event"], "evidence": source["evidence"],
                    "status": "ok", "verdict": final["verdict"], "label": LABELS[final["verdict"]],
                    "origin": "program_scope_guard" if "guard_reason" in final else "model",
                    "reason": final["reason"], "citations": citations, "assessment": final["assessment"],
                }
                if "model_verdict" in final:
                    item.update(model_verdict=final["model_verdict"], guard_reason=final["guard_reason"])
            items_by_id[event_id] = item
    items = [items_by_id[source["event_id"]] for source in retrieval["items"]]
    failed = [item["event_id"] for item in items if item["status"] == "error"]
    status = "error" if retrieval["status"] == "error" or (items and len(failed) == len(items)) else "partial" if failed or retrieval["status"] != "ok" else "ok"
    return {
        "schema_version": "agent-pipeline-v2.1-story-judge-v1", "stage": "judge", "status": status,
        "events": retrieval["events"], "items": items, "retrieval": retrieval,
        "batch_size": batch_size, "batch_reports": batch_reports,
        "failure_reasons": {"upstream_status": retrieval["status"], "failed_event_ids": failed},
    }


def build_story_report(judge: dict) -> dict:
    counts = {"consistent": 0, "contradiction": 0, "uncertain": 0, "failed": 0}
    findings = []
    for number, item in enumerate(judge["items"], 1):
        counts[item["verdict"]] += 1
        if item["status"] == "error":
            counts["failed"] += 1
        findings.append({
            "id": f"R{number}", "event_id": item["event_id"], "event_ids": [item["event_id"]],
            "actors": item["event"]["actors"], "event": item["event"]["event"],
            "verdict": item["verdict"], "label": item["label"], "reason": item["reason"],
            "citations": item["citations"], "status": item["status"], "origin": item["origin"],
        })
    verdict = "contradiction" if counts["contradiction"] else "uncertain" if counts["uncertain"] or counts["failed"] else "consistent"
    extraction = judge["retrieval"]["extraction"]
    return {
        "schema_version": "agent-pipeline-v2.1-story-report-v1", "stage": "report", "status": judge["status"],
        "processing_complete": judge["status"] == "ok",
        "summary": {"verdict": verdict, "counts": counts, "checked_event_count": len(judge["items"])},
        "findings": findings, "ignored_spans": extraction["ignored_spans"],
        "non_event_span_ids": extraction["non_event_span_ids"], "uncovered_span_ids": extraction.get("uncovered_span_ids", []),
        "judge": judge,
        "notice": "仅核对提取出的事件；不确定不是矛盾，未覆盖内容不代表无矛盾。",
    }


def process_story_v21(text: str, llm, retriever, method: str = "hybrid", metadata_filter: bool = False, top_k: int = 5, judge_batch_size: int = 8, device: str = "auto", progress=None) -> dict:
    extraction = extract_events(text, device=device, llm=llm, progress=progress)
    retrieval = retrieve_story_events(extraction, retriever, method, metadata_filter, top_k, progress)
    judge = judge_story_events(retrieval, llm, judge_batch_size, progress)
    return build_story_report(judge)

