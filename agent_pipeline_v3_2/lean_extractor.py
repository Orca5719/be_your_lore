"""Lean Extractor wire with canonical v2 output compatibility."""

from __future__ import annotations

import json
from pathlib import Path
import time

from agent_pipeline_v2.extractor import (
    SCHEMA_VERSION,
    _decode_window,
    _normalize_window_value,
    _windows,
    split_spans,
    validate_extraction,
)

PROMPT = (Path(__file__).parent / "prompts" / "lean_extractor_v1.txt").read_text(encoding="utf-8")
REQUIRED_EVENT_FIELDS = {"actors", "event", "modality", "source_ids", "context_ids", "check_reason"}
OPTIONAL_EVENT_FIELDS = {"mental_state", "conditions"}
TOP_LEVEL_FIELDS = {"events", "ignored_span_ids", "non_event_span_ids"}


def _to_legacy_wire(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_FIELDS:
        raise ValueError("Lean wire必须且只能包含events、ignored_span_ids、non_event_span_ids")
    if any(not isinstance(value[field], list) for field in TOP_LEVEL_FIELDS):
        raise ValueError("Lean wire三出口必须都是数组")
    events = []
    for raw in value["events"]:
        if not isinstance(raw, dict) or not REQUIRED_EVENT_FIELDS <= set(raw) or not set(raw) <= REQUIRED_EVENT_FIELDS | OPTIONAL_EVENT_FIELDS:
            raise ValueError("事件字段不符合Lean wire协议")
        modality = raw.get("modality")
        row = dict(raw)
        row.setdefault("mental_state", None)
        row.setdefault("conditions", [])
        row["explicit"] = modality != "inferred"
        events.append(row)
    ignored = value["ignored_span_ids"]
    if any(not isinstance(span_id, str) or not span_id.strip() for span_id in ignored) or len(ignored) != len(set(ignored)):
        raise ValueError("ignored_span_ids必须是不重复的文字数组")
    return {
        "events": events,
        "ignored_spans": [{"source_id": span_id, "reason": "process_detail"} for span_id in ignored],
        "non_event_span_ids": list(value["non_event_span_ids"]),
    }


def decode_lean_window(value: object, spans: dict[str, dict], window: dict, first_event_number: int):
    legacy = _to_legacy_wire(value)
    normalized, _ = _normalize_window_value(legacy, spans, window)
    return _decode_window(normalized, spans, window, first_event_number)


def _decode_with_changes(value: object, spans: dict[str, dict], window: dict, first_event_number: int):
    legacy = _to_legacy_wire(value)
    normalized, changes = _normalize_window_value(legacy, spans, window)
    return _decode_window(normalized, spans, window, first_event_number), normalized, changes


def _call_window(llm, messages: list[dict], spans: dict[str, dict], window: dict, first_event_number: int):
    attempts = []
    current = list(messages)
    normalizations: list[str] = []
    for attempt_number in range(2):
        raw = ""
        legacy_value = None
        try:
            llm.last_generation = {}
            raw = llm._generate(current, max_new_tokens=1536)
            value = json.loads(raw)
            legacy_value = _to_legacy_wire(value)
            legacy_value, changes = _normalize_window_value(legacy_value, spans, window)
            normalizations.extend(changes)
            decoded = _decode_window(legacy_value, spans, window, first_event_number)
            attempts.append({"attempt": attempt_number + 1, "status": "ok", "raw_output": raw, "timing": dict(llm.last_generation)})
            return decoded, {"status": "ok", "attempts": attempts, "wire_normalizations": list(dict.fromkeys(normalizations))}
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            error = str(exc)
            attempts.append({
                "attempt": attempt_number + 1, "status": "error", "raw_output": getattr(exc, "raw_output", raw),
                "error": error, "error_type": type(exc).__name__, "timing": dict(getattr(llm, "last_generation", {})),
            })
            if isinstance(exc, ValueError) and error.startswith("存在未覆盖的target_spans：") and isinstance(legacy_value, dict):
                missing = error.split("：", 1)[1].split(",")
                disposed = [span_id for span_id in window["target_ids"] if span_id not in missing]
                partial_window = {"target_ids": disposed, "context_ids": list(dict.fromkeys(window["context_ids"] + missing))}
                try:
                    base = _decode_window(legacy_value, spans, partial_window, first_event_number)
                    ordered = window["context_ids"] + window["target_ids"]
                    recovery_context = []
                    for span_id in missing:
                        position = ordered.index(span_id)
                        recovery_context.extend(ordered[max(0, position - 2):position])
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
                    recovery_decoded, _, recovery_changes = _decode_with_changes(
                        json.loads(recovery_raw), spans, recovery_window, first_event_number + len(base[0])
                    )
                    normalizations.extend(recovery_changes)
                    attempts.append({
                        "attempt": len(attempts) + 1, "purpose": "recover_missing_targets", "status": "ok",
                        "raw_output": recovery_raw, "timing": dict(llm.last_generation),
                    })
                    combined = (base[0] + recovery_decoded[0], base[1] + recovery_decoded[1], base[2] + recovery_decoded[2])
                    return combined, {
                        "status": "ok", "attempts": attempts,
                        "wire_normalizations": list(dict.fromkeys(normalizations)), "recovered_target_ids": missing,
                    }
                except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as recovery_exc:
                    attempts.append({
                        "attempt": len(attempts) + 1, "purpose": "recover_missing_targets", "status": "error",
                        "raw_output": locals().get("recovery_raw", ""), "error": str(recovery_exc),
                        "error_type": type(recovery_exc).__name__, "timing": dict(getattr(llm, "last_generation", {})),
                    })
            if attempt_number == 0 and isinstance(exc, (ValueError, json.JSONDecodeError)):
                current = current + [{
                    "role": "user",
                    "content": "上次回复不合格：" + error + "。重新处置本窗口全部target_spans，只返回Lean wire完整JSON对象。",
                }]
                continue
            break
    return None, {"status": "error", "attempts": attempts, "wire_normalizations": list(dict.fromkeys(normalizations))}


def extract_events_lean(text: str, device: str = "auto", llm=None, progress=None) -> dict:
    started = time.perf_counter()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device只能是auto/cpu/cuda")
    spans = split_spans(text)
    if len(text) > 800:
        raise ValueError("当前版本支持不超过800字符；不会截断")
    loaded_here = llm is None
    if llm is None:
        from agent_pipeline_v2.model import V2QwenJudge
        llm = V2QwenJudge(device)
    events, ignored, non_events, calls, rejected, failed_targets = [], [], [], [], [], []
    for window_id, window in enumerate(_windows(spans), 1):
        if progress:
            progress({"window_id": window_id, "purpose": "extract_checkable_events", "target_ids": window["target_ids"]})
        payload = {
            "target_spans": {span_id: spans[span_id]["text"] for span_id in window["target_ids"]},
            "context_spans": {span_id: spans[span_id]["text"] for span_id in window["context_ids"]},
        }
        messages = [
            {"role": "system", "content": PROMPT},
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
        "schema_version": SCHEMA_VERSION, "prompt_version": "extractor-lean-v1", "stage": "extraction", "status": status,
        "text": text, "spans": spans, "events": events, "ignored_spans": ignored,
        "non_event_span_ids": non_events, "uncovered_span_ids": sorted(failed_targets), "rejected": rejected,
        "calls": calls, "processing_complete": not failed_targets, "model_loaded_this_request": loaded_here,
        "device": getattr(llm, "device", device), "model_load_seconds": getattr(llm, "load_seconds", None),
        "request_seconds": time.perf_counter() - started,
    }
    validate_extraction(result)
    return result
