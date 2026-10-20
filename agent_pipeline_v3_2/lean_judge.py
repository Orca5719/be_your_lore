from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from agent_pipeline_v2.batch_llm import call_json_batch

LABELS = {"consistent": "明确吻合", "contradiction": "明确矛盾", "uncertain": "不确定"}
NONACTUAL = {"speech", "belief", "plan", "dream", "inferred"}
SCHEMA_VERSION = "agent-pipeline-v3.2-lean-judge-v1"
PROMPT = (Path(__file__).parent / "prompts" / "lean_judge_v1.txt").read_text(encoding="utf-8")


def validate_lean_answer(value: dict, evidence: list[dict]) -> None:
    if not isinstance(value, dict) or set(value) != {"verdict", "evidence_ids"}:
        raise ValueError("输出必须且只能包含verdict/evidence_ids")
    verdict = value["verdict"]
    if verdict not in LABELS:
        raise ValueError("verdict无效")
    ids = value["evidence_ids"]
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("evidence_ids必须是文字数组")
    if len(ids) != len(set(ids)):
        raise ValueError("evidence_ids不得重复")
    aliases = {f"L{number}" for number in range(1, len(evidence) + 1)}
    if any(item not in aliases for item in ids):
        raise ValueError("evidence_id不存在")
    if verdict != "uncertain" and not ids:
        raise ValueError("明确结论必须提供证据ID")


def finalize_lean_answer(value: dict, evidence: list[dict]) -> dict:
    validate_lean_answer(value, evidence)
    citations = [
        {"evidence_id": alias, "chunk_id": evidence[int(alias[1:]) - 1]["id"]}
        for alias in value["evidence_ids"]
    ]
    return {"verdict": value["verdict"], "evidence_ids": list(value["evidence_ids"]), "citations": citations}


def _messages(retrieval: dict, source: dict) -> list[dict]:
    lore = [
        {"evidence_id": f"L{number}", "text": chunk["text"], "heading_path": chunk.get("heading_path", [])}
        for number, chunk in enumerate(source["evidence"], 1)
    ]
    payload = {
        "fact": source.get("fact") or {"actors": source["event"].get("actors", []), "normalized_fact": source["event"].get("event")},
        "story_context": retrieval.get("extraction", {}).get("text", ""),
        "lore": lore,
    }
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _uncertain(source: dict, origin: str, status: str = "ok") -> dict:
    return {
        "event_id": source["event_id"], "event": source["event"], "evidence": source.get("evidence", []),
        "status": status, "verdict": "uncertain", "label": LABELS["uncertain"],
        "origin": origin, "evidence_ids": [], "citations": [],
    }


def judge_frozen_retrieval(retrieval: dict, llm, batch_size: int = 8, progress=None) -> dict:
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
    batch_reports = []
    for offset in range(0, len(pending), batch_size):
        group = pending[offset:offset + batch_size]
        if progress:
            progress({"purpose": "lean_judge", "window_id": len(batch_reports) + 1, "target_ids": [x["event_id"] for x in group]})
        requests = [{"request_id": source["event_id"], "messages": _messages(retrieval, source)} for source in group]
        validators = [(lambda evidence: lambda value: validate_lean_answer(value, evidence))(source["evidence"]) for source in group]
        values, batch_report = call_json_batch(llm, requests, max_output=128, validators=validators)
        batch_reports.append(batch_report)
        rows = {row["request_id"]: row for row in batch_report["rows"]}
        for source, value in zip(group, values):
            event_id = source["event_id"]
            if value is None:
                item = _uncertain(source, "judge_failure", "error")
                item["error"] = rows[event_id].get("error")
            else:
                final = finalize_lean_answer(value, source["evidence"])
                item = {
                    "event_id": event_id, "event": source["event"], "evidence": source["evidence"],
                    "status": "ok", "verdict": final["verdict"], "label": LABELS[final["verdict"]],
                    "origin": "model", "evidence_ids": final["evidence_ids"], "citations": final["citations"],
                }
            items_by_id[event_id] = item
    items = [items_by_id[source["event_id"]] for source in retrieval.get("items", [])]
    failed = [item["event_id"] for item in items if item["status"] == "error"]
    upstream = retrieval.get("status", "error")
    status = "error" if upstream == "error" or (items and len(failed) == len(items)) else "partial" if failed or upstream != "ok" else "ok"
    return {
        "schema_version": SCHEMA_VERSION, "stage": "judge", "status": status,
        "events": copy.deepcopy(retrieval.get("events", [])), "items": items,
        "retrieval": copy.deepcopy(retrieval), "batch_size": batch_size,
        "batch_reports": batch_reports, "request_seconds": time.perf_counter() - started,
        "failure_reasons": {"upstream_status": upstream, "failed_event_ids": failed},
    }
