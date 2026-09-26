"""One-stage extraction of worldbuilding-relevant story events."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time


SCHEMA_VERSION = "agent-pipeline-v2-extraction-v1"
MODALITIES = {"observed", "speech", "belief", "plan", "dream", "conditional", "inferred"}
CHECK_REASONS = {"mechanism", "knowledge_relation", "state_time_space", "consequence_support"}
IGNORE_REASONS = {"routine", "process_detail"}
EVENT_FIELDS = {
    "id",
    "actors",
    "event",
    "mental_state",
    "explicit",
    "modality",
    "conditions",
    "source_ids",
    "context_ids",
    "check_reason",
}


def split_spans(text: str) -> dict[str, dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("输入不能为空")
    spans: dict[str, dict] = {}
    for match in re.finditer(r"[^\r\n，,。！？!?；;]+[，,。！？!?；;]?", text):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        if start == end:
            continue
        for segment_start in range(start, end, 180):
            segment_end = min(segment_start + 180, end)
            span_id = f"S{len(spans) + 1}"
            spans[span_id] = {
                "text": text[segment_start:segment_end],
                "start": segment_start,
                "end": segment_end,
            }
    if not spans:
        raise ValueError("没有可提取的文字片段")
    return spans


def _string_list(value, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field}须为文字数组")
    if nonempty and not value:
        raise ValueError(f"{field}不能为空")
    if len(value) != len(set(value)):
        raise ValueError(f"{field}不能重复")
    return value


def _validate_event(row: dict, spans: dict[str, dict], seen_ids: set[str]) -> None:
    if not isinstance(row, dict) or set(row) != EVENT_FIELDS:
        raise ValueError("事件字段不符合v2协议")
    event_id = row["id"]
    if not isinstance(event_id, str) or not event_id.strip() or event_id in seen_ids:
        raise ValueError("事件ID无效或重复")
    seen_ids.add(event_id)
    actors = _string_list(row["actors"], "actors", nonempty=True)
    source_ids = _string_list(row["source_ids"], "source_ids", nonempty=True)
    context_ids = _string_list(row["context_ids"], "context_ids")
    _string_list(row["conditions"], "conditions")
    if any(span_id not in spans for span_id in source_ids + context_ids):
        raise ValueError("事件引用了不存在的原文片段")
    if set(source_ids) & set(context_ids):
        raise ValueError("source_ids和context_ids不能重叠")
    if not isinstance(row["event"], str) or not row["event"].strip():
        raise ValueError("event不能为空")
    if row["mental_state"] is not None and (
        not isinstance(row["mental_state"], str) or not row["mental_state"].strip()
    ):
        raise ValueError("mental_state须为文字或null")
    if type(row["explicit"]) is not bool or row["modality"] not in MODALITIES:
        raise ValueError("事件叙述性质无效")
    if (row["modality"] == "inferred") == row["explicit"]:
        raise ValueError("inferred必须explicit=false，其他modality必须explicit=true")
    if row["check_reason"] not in CHECK_REASONS:
        raise ValueError("核对原因无效")
    referenced_text = "".join(spans[span_id]["text"] for span_id in source_ids + context_ids)
    if any(actor not in referenced_text for actor in actors):
        raise ValueError("主体缺少原文依据")


def validate_extraction(report: dict) -> dict[str, dict]:
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("提取报告schema_version无效")
    if report.get("stage") != "extraction" or report.get("status") not in {"ok", "partial", "error"}:
        raise ValueError("提取报告阶段或状态无效")
    text = report.get("text")
    spans = report.get("spans")
    if not isinstance(text, str) or not text.strip() or not isinstance(spans, dict) or not spans:
        raise ValueError("提取报告缺少原文或spans")
    for span_id, span in spans.items():
        if (
            not isinstance(span_id, str)
            or not isinstance(span, dict)
            or set(span) != {"text", "start", "end"}
            or not isinstance(span["text"], str)
            or type(span["start"]) is not int
            or type(span["end"]) is not int
            or span["text"] != text[span["start"] : span["end"]]
        ):
            raise ValueError("span来源定位无效")

    events = report.get("events")
    ignored = report.get("ignored_spans")
    non_events = report.get("non_event_span_ids")
    if not isinstance(events, list) or not isinstance(ignored, list) or not isinstance(non_events, list):
        raise ValueError("三出口必须都是数组")
    seen_event_ids: set[str] = set()
    disposition: dict[str, dict] = {}

    def assign(span_id: str, value: dict) -> None:
        if span_id not in spans:
            raise ValueError("处置引用了不存在的原文片段")
        if span_id in disposition:
            raise ValueError("同一片段出现重复处置")
        disposition[span_id] = value

    for row in events:
        _validate_event(row, spans, seen_event_ids)
        for span_id in row["source_ids"]:
            if span_id not in disposition:
                disposition[span_id] = {"disposition": "event", "event_ids": []}
            elif disposition[span_id]["disposition"] != "event":
                raise ValueError("同一片段出现重复处置")
            disposition[span_id]["event_ids"].append(row["id"])
    seen_ignored: set[str] = set()
    for row in ignored:
        if not isinstance(row, dict) or set(row) != {"source_id", "reason"}:
            raise ValueError("ignored_spans字段无效")
        if row["reason"] not in IGNORE_REASONS:
            raise ValueError("忽略原因无效")
        if row["source_id"] in seen_ignored:
            raise ValueError("同一片段出现重复处置")
        seen_ignored.add(row["source_id"])
        assign(row["source_id"], {"disposition": "ignored", "reason": row["reason"]})
    _string_list(non_events, "non_event_span_ids")
    for span_id in non_events:
        assign(span_id, {"disposition": "non_event"})
    missing = sorted(set(spans) - set(disposition))
    if missing:
        if report["status"] == "ok" or report.get("uncovered_span_ids") != missing:
            raise ValueError("存在未覆盖的原文片段：" + ",".join(missing))
    if report["status"] == "ok" and report.get("rejected"):
        raise ValueError("ok报告不能包含rejected")
    return disposition


def _windows(spans: dict[str, dict]) -> list[dict]:
    span_ids = list(spans)
    result = []
    target_ids: list[str] = []
    chars = 0
    for span_id in span_ids:
        size = len(spans[span_id]["text"])
        if target_ids and (len(target_ids) >= 8 or chars + size > 180):
            first = span_ids.index(target_ids[0])
            result.append({"target_ids": target_ids, "context_ids": span_ids[max(0, first - 2) : first]})
            target_ids = []
            chars = 0
        target_ids.append(span_id)
        chars += size
    if target_ids:
        first = span_ids.index(target_ids[0])
        result.append({"target_ids": target_ids, "context_ids": span_ids[max(0, first - 2) : first]})
    return result


def _normalize_window_value(value: object, spans: dict[str, dict], window: dict) -> tuple[object, list[str]]:
    """Repair a small set of unambiguous wire-shape mistakes before strict validation."""
    if not isinstance(value, dict):
        return value, []
    normalized = {
        key: [dict(row) if isinstance(row, dict) else row for row in rows] if isinstance(rows, list) else rows
        for key, rows in value.items()
    }
    changes: list[str] = []
    events = normalized.get("events")
    ignored = normalized.get("ignored_spans")
    non_events = normalized.get("non_event_span_ids")
    targets = set(window["target_ids"])
    contexts = set(window["context_ids"])

    if isinstance(events, list):
        ordered_ids = window["context_ids"] + window["target_ids"]
        positions = {span_id: index for index, span_id in enumerate(ordered_ids)}
        for row in events:
            if not isinstance(row, dict):
                continue
            for field in ("actors", "conditions", "source_ids", "context_ids"):
                if isinstance(row.get(field), str) and row[field].strip():
                    row[field] = [row[field]]
                    changes.append(f"scalar_{field}_to_list")
            actors = row.get("actors")
            source_ids = row.get("source_ids")
            context_ids = row.get("context_ids")
            if not (isinstance(actors, list) and isinstance(source_ids, list) and isinstance(context_ids, list)):
                continue
            referenced = [span_id for span_id in source_ids + context_ids if span_id in spans]
            source_positions = [positions[span_id] for span_id in source_ids if span_id in positions]
            limit = min(source_positions) if source_positions else len(ordered_ids)
            for actor in actors:
                if not isinstance(actor, str) or any(actor in spans[span_id]["text"] for span_id in referenced):
                    continue
                candidates = [
                    span_id for span_id in ordered_ids[:limit]
                    if actor in spans[span_id]["text"] and span_id not in source_ids and span_id not in context_ids
                ]
                if candidates:
                    context_ids.append(candidates[-1])
                    referenced.append(candidates[-1])
                    changes.append("actor_context_added")

        kept_events = []
        for row in events:
            source_ids = row.get("source_ids") if isinstance(row, dict) else None
            if (
                isinstance(source_ids, list)
                and source_ids
                and all(span_id in contexts for span_id in source_ids)
                and not any(span_id in targets for span_id in source_ids)
            ):
                changes.append("context_event_removed")
                continue
            kept_events.append(row)
        normalized["events"] = kept_events
        events = kept_events

    event_sources = {
        span_id
        for row in events if isinstance(events, list) and isinstance(row, dict)
        for span_id in row.get("source_ids", []) if isinstance(row.get("source_ids"), list)
    } if isinstance(events, list) else set()
    ignored_ids: set[str] = set()
    if isinstance(ignored, list):
        kept = []
        for row in ignored:
            span_id = row.get("source_id") if isinstance(row, dict) else None
            if span_id in contexts and span_id not in targets:
                changes.append("context_disposition_removed")
                continue
            if span_id in event_sources:
                changes.append("duplicate_ignored_removed")
                continue
            if span_id in ignored_ids:
                changes.append("duplicate_ignored_removed")
                continue
            kept.append(row)
            if isinstance(span_id, str):
                ignored_ids.add(span_id)
        normalized["ignored_spans"] = kept
    if isinstance(non_events, list):
        kept = []
        seen: set[str] = set()
        for span_id in non_events:
            if span_id in contexts and span_id not in targets:
                changes.append("context_disposition_removed")
                continue
            if span_id in event_sources or span_id in ignored_ids or span_id in seen:
                changes.append("duplicate_non_event_removed")
                continue
            kept.append(span_id)
            if isinstance(span_id, str):
                seen.add(span_id)
        normalized["non_event_span_ids"] = kept
    return normalized, list(dict.fromkeys(changes))


def _decode_window(value: dict, spans: dict[str, dict], window: dict, first_event_number: int) -> tuple[list[dict], list[dict], list[str]]:
    if not isinstance(value, dict) or set(value) != {"events", "ignored_spans", "non_event_span_ids"}:
        raise ValueError("输出必须且只能包含events、ignored_spans、non_event_span_ids")
    if not all(isinstance(value[field], list) for field in value):
        raise ValueError("三出口必须都是数组")
    targets = set(window["target_ids"])
    allowed_context = set(window["context_ids"])
    events = []
    disposition: dict[str, str] = {}
    seen_ids: set[str] = set()
    wire_fields = EVENT_FIELDS - {"id"}
    for index, raw in enumerate(value["events"], first_event_number):
        if not isinstance(raw, dict) or set(raw) != wire_fields:
            raise ValueError("事件字段不符合v2 wire协议")
        row = {"id": f"E{index}", **raw}
        if any(span_id not in targets for span_id in row["source_ids"]):
            raise ValueError("source_ids只能引用当前target_spans")
        if any(span_id not in targets | allowed_context for span_id in row["context_ids"]):
            raise ValueError("context_ids超出当前窗口")
        _validate_event(row, spans, seen_ids)
        for span_id in row["source_ids"]:
            if span_id in disposition and disposition[span_id] != "event":
                raise ValueError("同一片段出现重复处置")
            disposition[span_id] = "event"
        events.append(row)
    ignored = []
    for raw in value["ignored_spans"]:
        if not isinstance(raw, dict) or set(raw) != {"source_id", "reason"}:
            raise ValueError("ignored_spans字段无效")
        span_id = raw["source_id"]
        if span_id not in targets or raw["reason"] not in IGNORE_REASONS:
            raise ValueError("忽略编号或忽略原因无效")
        if span_id in disposition:
            raise ValueError("同一片段出现重复处置")
        disposition[span_id] = "ignored"
        ignored.append(raw)
    non_events = _string_list(value["non_event_span_ids"], "non_event_span_ids")
    for span_id in non_events:
        if span_id not in targets:
            raise ValueError("non_event_span_ids只能引用当前target_spans")
        if span_id in disposition:
            raise ValueError("同一片段出现重复处置")
        disposition[span_id] = "non_event"
    missing = sorted(targets - set(disposition))
    if missing:
        raise ValueError("存在未覆盖的target_spans：" + ",".join(missing))
    return events, ignored, non_events


def _call_window(llm, messages: list[dict], spans: dict[str, dict], window: dict, first_event_number: int):
    attempts = []
    current = list(messages)
    wire_normalizations: list[str] = []
    for attempt in range(2):
        raw = ""
        value = None
        try:
            llm.last_generation = {}
            raw = llm._generate(current, max_new_tokens=1536)
            value = json.loads(raw)
            value, changes = _normalize_window_value(value, spans, window)
            wire_normalizations.extend(changes)
            decoded = _decode_window(value, spans, window, first_event_number)
            attempts.append({"attempt": attempt + 1, "status": "ok", "raw_output": raw, "timing": dict(llm.last_generation)})
            return decoded, {"status": "ok", "attempts": attempts, "wire_normalizations": list(dict.fromkeys(wire_normalizations))}
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            error = str(exc)
            attempts.append({"attempt": attempt + 1, "status": "error", "raw_output": getattr(exc, "raw_output", raw), "error": error, "error_type": type(exc).__name__, "timing": dict(getattr(llm, "last_generation", {}))})
            if isinstance(exc, ValueError) and error.startswith("存在未覆盖的target_spans：") and isinstance(value, dict):
                missing = error.split("：", 1)[1].split(",")
                disposed = [span_id for span_id in window["target_ids"] if span_id not in missing]
                partial_window = {
                    "target_ids": disposed,
                    "context_ids": list(dict.fromkeys(window["context_ids"] + missing)),
                }
                try:
                    base = _decode_window(value, spans, partial_window, first_event_number)
                    ordered = window["context_ids"] + window["target_ids"]
                    recovery_context = []
                    for span_id in missing:
                        position = ordered.index(span_id)
                        recovery_context.extend(ordered[max(0, position - 2) : position])
                    recovery_window = {
                        "target_ids": missing,
                        "context_ids": [span_id for span_id in dict.fromkeys(recovery_context) if span_id not in missing],
                    }
                    payload = {
                        "target_spans": {span_id: spans[span_id]["text"] for span_id in recovery_window["target_ids"]},
                        "context_spans": {span_id: spans[span_id]["text"] for span_id in recovery_window["context_ids"]},
                    }
                    llm.last_generation = {}
                    recovery_raw = llm._generate(
                        [messages[0], {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                        max_new_tokens=1536,
                    )
                    recovery_value = json.loads(recovery_raw)
                    recovery_value, recovery_changes = _normalize_window_value(recovery_value, spans, recovery_window)
                    wire_normalizations.extend(recovery_changes)
                    recovered = _decode_window(recovery_value, spans, recovery_window, first_event_number + len(base[0]))
                    attempts.append({"attempt": len(attempts) + 1, "purpose": "recover_missing_targets", "status": "ok", "raw_output": recovery_raw, "timing": dict(llm.last_generation)})
                    combined = (base[0] + recovered[0], base[1] + recovered[1], base[2] + recovered[2])
                    return combined, {
                        "status": "ok",
                        "attempts": attempts,
                        "wire_normalizations": list(dict.fromkeys(wire_normalizations)),
                        "recovered_target_ids": missing,
                    }
                except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as recovery_exc:
                    attempts.append({"attempt": len(attempts) + 1, "purpose": "recover_missing_targets", "status": "error", "raw_output": locals().get("recovery_raw", ""), "error": str(recovery_exc), "error_type": type(recovery_exc).__name__, "timing": dict(getattr(llm, "last_generation", {}))})
            if attempt == 0 and isinstance(exc, (ValueError, json.JSONDecodeError)):
                current = current + [{"role": "user", "content": "上次回复不合格：" + error + "。请重新审计本窗口的全部target_spans，只返回符合协议的完整JSON对象。"}]
                continue
            break
    return None, {"status": "error", "attempts": attempts, "wire_normalizations": list(dict.fromkeys(wire_normalizations))}


def extract_events(text: str, device: str = "auto", llm=None, progress=None) -> dict:
    started = time.perf_counter()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device只能是auto/cpu/cuda")
    spans = split_spans(text)
    if len(text) > 800:
        raise ValueError("当前版本支持不超过800字符；不会截断")
    loaded_here = llm is None
    if llm is None:
        from .model import V2QwenJudge

        llm = V2QwenJudge(device)
    prompt = (Path(__file__).parent / "prompts" / "extractor_v1.txt").read_text(encoding="utf-8")
    events: list[dict] = []
    ignored: list[dict] = []
    non_events: list[str] = []
    calls = []
    rejected = []
    failed_targets: list[str] = []
    for window_id, window in enumerate(_windows(spans), 1):
        if progress:
            progress({"window_id": window_id, "purpose": "extract_checkable_events", "target_ids": window["target_ids"]})
        payload = {
            "target_spans": {span_id: spans[span_id]["text"] for span_id in window["target_ids"]},
            "context_spans": {span_id: spans[span_id]["text"] for span_id in window["context_ids"]},
        }
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ]
        decoded, call = _call_window(llm, messages, spans, window, len(events) + 1)
        call.update({"window_id": window_id, "purpose": "extract_checkable_events", **window})
        calls.append(call)
        if decoded is None:
            failed_targets.extend(window["target_ids"])
            rejected.append({"window_id": window_id, "target_ids": window["target_ids"], "error": call["attempts"][-1]["error"]})
            continue
        window_events, window_ignored, window_non_events = decoded
        events.extend(window_events)
        ignored.extend(window_ignored)
        non_events.extend(window_non_events)
    status = "ok" if not failed_targets else "error" if len(failed_targets) == len(spans) else "partial"
    result = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": "extractor-v1",
        "stage": "extraction",
        "status": status,
        "text": text,
        "spans": spans,
        "events": events,
        "ignored_spans": ignored,
        "non_event_span_ids": non_events,
        "uncovered_span_ids": sorted(failed_targets),
        "rejected": rejected,
        "calls": calls,
        "processing_complete": not failed_targets,
        "model_loaded_this_request": loaded_here,
        "device": getattr(llm, "device", device),
        "model_load_seconds": getattr(llm, "load_seconds", None),
        "request_seconds": time.perf_counter() - started,
    }
    validate_extraction(result)
    return result
