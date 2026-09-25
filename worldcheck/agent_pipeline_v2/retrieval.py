"""Embedding-only retrieval for v2 extracted events."""

from __future__ import annotations

import copy
from pathlib import Path
import time

from .extractor import validate_extraction


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "agent-pipeline-v2-retrieval-v1"


def event_query(event: dict, spans: dict[str, dict]) -> str:
    parts = [
        "主体：" + "、".join(event["actors"]),
        "核对维度：" + event["check_reason"],
        "叙述性质：" + event["modality"],
        "事件：" + event["event"],
    ]
    if event.get("mental_state"):
        parts.append("心理：" + event["mental_state"])
    if event["conditions"]:
        parts.append("条件：" + "；".join(event["conditions"]))
    ordered = sorted(
        dict.fromkeys(event["context_ids"] + event["source_ids"]),
        key=lambda span_id: spans[span_id]["start"],
    )
    if ordered:
        parts.append("局部原文：" + "".join(spans[span_id]["text"] for span_id in ordered))
    return "\n".join(parts)


def validate_retrieval(report: dict) -> None:
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v2检索报告schema_version无效")
    if report.get("stage") != "retrieval" or report.get("status") not in {"ok", "partial", "error"}:
        raise ValueError("v2检索报告阶段或状态无效")
    extraction = report.get("extraction")
    validate_extraction(extraction)
    if report.get("events") != extraction["events"]:
        raise ValueError("检索事件与提取事件不匹配")
    items = report.get("items")
    if not isinstance(items, list):
        raise ValueError("检索items须为数组")
    events = {row["id"]: row for row in extraction["events"]}
    seen = set()
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("event_id") not in events
            or item["event_id"] in seen
            or item.get("event") != events[item["event_id"]]
            or item.get("status") not in {"ok", "error"}
        ):
            raise ValueError("检索item无效或重复")
        seen.add(item["event_id"])
        if not isinstance(item.get("evidence"), list):
            raise ValueError("evidence须为数组")
        evidence_ids = set()
        for chunk in item["evidence"]:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("id"), str)
                or not isinstance(chunk.get("text"), str)
                or chunk["id"] in evidence_ids
            ):
                raise ValueError("证据ID或原文无效")
            evidence_ids.add(chunk["id"])
    if seen != set(events):
        raise ValueError("检索报告缺少提取事件")


def retrieve_events(
    extraction_report: dict,
    device: str = "auto",
    index_directory=None,
    k: int = 5,
    retriever=None,
    progress=None,
) -> dict:
    validate_extraction(extraction_report)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k必须为正整数")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device必须为auto/cpu/cuda")
    started = time.perf_counter()
    upstream = copy.deepcopy(extraction_report)
    loaded_here = False
    load_error = None
    index_info = None
    if upstream["events"] and upstream["status"] != "error" and retriever is None:
        try:
            from encoder import Encoder
            from retrieval import Retriever

            path = Path(index_directory) if index_directory is not None else ROOT / "data" / "index"
            encoder = Encoder(device=device, offline=True, precision="float32")
            retriever = Retriever(path, encoder)
            loaded_here = True
        except (ValueError, OSError, RuntimeError) as exc:
            load_error = str(exc)
    if retriever is not None and hasattr(retriever, "metadata"):
        metadata = retriever.metadata
        index_info = {key: copy.deepcopy(metadata[key]) for key in ("config", "sources", "lore_directory") if key in metadata}
    items = []
    cache: dict[str, list] = {}
    for event in upstream["events"]:
        event_id = event["id"]
        query = event_query(event, upstream["spans"])
        if progress:
            progress({"window_id": len(items) + 1, "purpose": "retrieval", "target_ids": [event_id]})
        item = {"event_id": event_id, "event": event, "query": query, "evidence": [], "cache_hit": query in cache}
        item_started = time.perf_counter()
        try:
            if load_error:
                raise ValueError(load_error)
            if retriever is None:
                raise ValueError("检索器不可用")
            if query not in cache:
                cache[query] = retriever.search(query, k=k)
            item.update(status="ok", evidence=copy.deepcopy(cache[query]))
        except (ValueError, OSError, RuntimeError) as exc:
            item.update(status="error", error=str(exc))
        item["retrieval_seconds"] = time.perf_counter() - item_started
        items.append(item)
    failures = [item["event_id"] for item in items if item["status"] == "error"]
    if upstream["status"] == "error" or (items and len(failures) == len(items)):
        status = "error"
    elif failures or upstream["status"] != "ok":
        status = "partial"
    else:
        status = "ok"
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "retrieval",
        "status": status,
        "extraction": upstream,
        "events": upstream["events"],
        "items": items,
        "top_k": k,
        "upstream_status": upstream["status"],
        "failure_reasons": {"upstream_status": upstream["status"], "failed_event_ids": failures, "load_error": load_error},
        "retrieval_complete": not failures and upstream["status"] != "error",
        "processing_complete": status == "ok",
        "encoder_loaded_this_request": loaded_here,
        "index": index_info,
        "device": getattr(getattr(retriever, "encoder", None), "device", device),
        "request_seconds": time.perf_counter() - started,
    }
    validate_retrieval(result)
    return result
