"""Lean Extractor wire with canonical v2 output compatibility."""

from __future__ import annotations

import json
import hashlib
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

PROMPT = (Path(__file__).parent / "prompts" / "lean_extractor_v2.txt").read_text(encoding="utf-8")
DISPOSITION_PROMPT = (
    "只处理一个小说片段，并返回严格JSON：{\"disposition\":\"ignored|non_event|event\"}。"
    "ignored表示片段有独立内容，但只是普通场景、动作或暂时状态，不形成世界观约束或持续后果；"
    "non_event表示片段没有独立命题；event表示片段可能受世界观约束或改变后续故事，"
    "此时不要编造事件内容。拿不准时选event。只依据提供的原文和上下文。"
)
PROMPT_SHA256 = hashlib.sha256((PROMPT + "\n" + DISPOSITION_PROMPT).encode("utf-8")).hexdigest()
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


def _sanitize_recovery_value(value: object, recovery_window: dict) -> tuple[object, list[str]]:
    """Drop only out-of-scope disposition IDs from a recovery response.

    Recovery prompts ask the model to classify a small set of missing targets.  The
    model can repeat an already handled span in one of the two disposition lists.
    That repetition must not invalidate an otherwise useful recovery response.
    Event references remain untouched and continue through strict validation.
    """
    if not isinstance(value, dict):
        return value, []
    target_ids = set(recovery_window.get("target_ids", []))
    sanitized = dict(value)
    changed = False
    for field in ("ignored_span_ids", "non_event_span_ids"):
        raw_ids = value.get(field)
        if not isinstance(raw_ids, list):
            continue
        filtered = [span_id for span_id in raw_ids if span_id in target_ids]
        if filtered != raw_ids:
            sanitized[field] = filtered
            changed = True
    return sanitized, ["recovery_out_of_scope_disposition_removed"] if changed else []


def _recover_disposition(llm, spans: dict[str, dict], window: dict, span_id: str) -> tuple[str, str]:
    target_ids = window["target_ids"]
    position = target_ids.index(span_id)
    neighbors = target_ids[max(0, position - 2):position] + target_ids[position + 1:position + 3]
    payload = {
        "target_span": {span_id: spans[span_id]["text"]},
        "context_spans": {item: spans[item]["text"] for item in neighbors},
    }
    raw = llm._generate([
        {"role": "system", "content": DISPOSITION_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ], max_new_tokens=32)
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"disposition"} or value["disposition"] not in {"ignored", "non_event", "event"}:
        error = ValueError("单片段处置必须是ignored、non_event或event")
        error.raw_output = raw
        raise error
    return value["disposition"], raw


def _call_window(llm, messages: list[dict], spans: dict[str, dict], window: dict, first_event_number: int):
    attempts = []
    current = list(messages)
    normalizations: list[str] = []
    best_partial = None
    best_missing = list(window["target_ids"])
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
                    if len(missing) < len(best_missing):
                        best_partial, best_missing = base, missing
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
                    recovered = ([], [], [])
                    for recovery_round in range(2):
                        llm.last_generation = {}
                        recovery_raw = llm._generate(
                            [messages[0], {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                            max_new_tokens=1536,
                        )
                        recovery_value, sanitize_changes = _sanitize_recovery_value(
                            json.loads(recovery_raw), recovery_window
                        )
                        normalizations.extend(sanitize_changes)
                        legacy_recovery = _to_legacy_wire(recovery_value)
                        normalized_recovery, recovery_changes = _normalize_window_value(
                            legacy_recovery, spans, recovery_window
                        )
                        normalizations.extend(recovery_changes)
                        try:
                            recovery_decoded = _decode_window(
                                normalized_recovery, spans, recovery_window,
                                first_event_number + len(base[0]) + len(recovered[0]),
                            )
                            remaining = []
                        except ValueError as coverage_exc:
                            coverage_error = str(coverage_exc)
                            if not coverage_error.startswith("存在未覆盖的target_spans：") or recovery_round == 1:
                                raise
                            remaining = coverage_error.split("：", 1)[1].split(",")
                            handled = [span_id for span_id in recovery_window["target_ids"] if span_id not in remaining]
                            recovery_decoded = _decode_window(
                                normalized_recovery, spans,
                                {"target_ids": handled, "context_ids": list(dict.fromkeys(recovery_window["context_ids"] + remaining))},
                                first_event_number + len(base[0]) + len(recovered[0]),
                            )
                        recovered = tuple(left + right for left, right in zip(recovered, recovery_decoded))
                        if len(remaining) < len(best_missing):
                            best_partial = tuple(left + right for left, right in zip(base, recovered))
                            best_missing = remaining
                        attempts.append({
                            "attempt": len(attempts) + 1, "purpose": "recover_missing_targets", "status": "ok",
                            "raw_output": recovery_raw, "timing": dict(llm.last_generation),
                            "remaining_target_ids": remaining,
                        })
                        if not remaining:
                            break
                        recovery_window = {
                            "target_ids": remaining,
                            "context_ids": list(dict.fromkeys(recovery_window["context_ids"] + handled)),
                        }
                        payload = {
                            "target_spans": {span_id: spans[span_id]["text"] for span_id in remaining},
                            "context_spans": {span_id: spans[span_id]["text"] for span_id in recovery_window["context_ids"]},
                            "coverage_instruction": "必须将每个target_span归入events、ignored_span_ids或non_event_span_ids，不能遗漏。",
                        }
                    combined = tuple(left + right for left, right in zip(base, recovered))
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
    disposition_resolved = []
    if best_partial is not None and best_missing:
        unresolved = []
        for span_id in best_missing:
            raw = ""
            try:
                llm.last_generation = {}
                disposition, raw = _recover_disposition(llm, spans, window, span_id)
                attempts.append({
                    "attempt": len(attempts) + 1, "purpose": "recover_disposition", "status": "ok" if disposition != "event" else "error",
                    "target_id": span_id, "disposition": disposition, "raw_output": raw,
                    "error": "片段仍需提取事件" if disposition == "event" else None,
                    "timing": dict(llm.last_generation),
                })
                if disposition == "ignored":
                    best_partial[1].append({"source_id": span_id, "reason": "process_detail"})
                    disposition_resolved.append(span_id)
                elif disposition == "non_event":
                    best_partial[2].append(span_id)
                    disposition_resolved.append(span_id)
                else:
                    unresolved.append(span_id)
            except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as disposition_exc:
                attempts.append({
                    "attempt": len(attempts) + 1, "purpose": "recover_disposition", "status": "error",
                    "target_id": span_id, "raw_output": getattr(disposition_exc, "raw_output", raw), "error": str(disposition_exc),
                    "error_type": type(disposition_exc).__name__, "timing": dict(getattr(llm, "last_generation", {})),
                })
                unresolved.append(span_id)
        best_missing = unresolved
    if best_partial is not None and not best_missing:
        return best_partial, {
            "status": "ok", "attempts": attempts,
            "wire_normalizations": list(dict.fromkeys(normalizations)),
            "disposition_recovered_target_ids": disposition_resolved,
        }
    if best_partial is not None:
        return best_partial, {
            "status": "partial", "attempts": attempts,
            "wire_normalizations": list(dict.fromkeys(normalizations)),
            "uncovered_target_ids": best_missing,
        }
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
        if call["status"] == "partial":
            failed_targets.extend(call["uncovered_target_ids"])
            rejected.append({
                "window_id": window_id, "target_ids": call["uncovered_target_ids"],
                "error": "恢复后仍有未覆盖的target_spans：" + ",".join(call["uncovered_target_ids"]),
            })
        window_events, window_ignored, window_non_events = decoded
        events.extend(window_events)
        ignored.extend(window_ignored)
        non_events.extend(window_non_events)
    status = "ok" if not failed_targets else "error" if len(failed_targets) == len(spans) else "partial"
    result = {
        "schema_version": SCHEMA_VERSION, "prompt_version": "extractor-lean-v2", "stage": "extraction", "status": status,
        "text": text, "spans": spans, "events": events, "ignored_spans": ignored,
        "non_event_span_ids": non_events, "uncovered_span_ids": sorted(failed_targets), "rejected": rejected,
        "calls": calls, "processing_complete": not failed_targets, "model_loaded_this_request": loaded_here,
        "device": getattr(llm, "device", device), "model_load_seconds": getattr(llm, "load_seconds", None),
        "request_seconds": time.perf_counter() - started,
    }
    validate_extraction(result)
    return result
