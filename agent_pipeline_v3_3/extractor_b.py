"""Benchmark 3.3B: accept valid partial extraction, account for missing spans in code."""

from __future__ import annotations

import json
from pathlib import Path
import time

from agent_pipeline_v2.extractor import (
    SCHEMA_VERSION, _decode_window, _normalize_window_value, _windows,
    split_spans, validate_extraction,
)

from .coverage import account_coverage


PROMPT_PATH = Path(__file__).resolve().parents[1] / "agent_pipeline_v2" / "prompts" / "extractor_v1.txt"


def _decode_partial(value, spans, window, first_event_number):
    """Validate every supplied disposition, leaving only unmentioned targets pending."""
    normalized, changes = _normalize_window_value(value, spans, window)
    if not isinstance(normalized, dict):
        raise ValueError("提取输出必须是JSON对象")
    target_ids = window["target_ids"]
    mentioned = set()
    for row in normalized.get("events", []) if isinstance(normalized.get("events"), list) else []:
        if isinstance(row, dict) and isinstance(row.get("source_ids"), list):
            mentioned.update(span_id for span_id in row["source_ids"] if isinstance(span_id, str))
    for row in normalized.get("ignored_spans", []) if isinstance(normalized.get("ignored_spans"), list) else []:
        if isinstance(row, dict) and isinstance(row.get("source_id"), str):
            mentioned.add(row["source_id"])
    if isinstance(normalized.get("non_event_span_ids"), list):
        mentioned.update(span_id for span_id in normalized["non_event_span_ids"] if isinstance(span_id, str))
    missing = [span_id for span_id in target_ids if span_id not in mentioned]
    subwindow = {
        "target_ids": [span_id for span_id in target_ids if span_id in mentioned],
        "context_ids": list(dict.fromkeys(window["context_ids"] + missing)),
    }
    decoded = _decode_window(normalized, spans, subwindow, first_event_number)
    accounted = account_coverage(target_ids, *decoded)
    if accounted["uncovered_span_ids"] != missing:
        raise ValueError("程序覆盖账本与输出处置不一致")
    return decoded, missing, changes


def _recovery_window(spans, original, missing):
    ordered = original["context_ids"] + original["target_ids"]
    context = []
    for span_id in missing:
        position = ordered.index(span_id)
        context.extend(ordered[max(0, position - 2):position])
    return {
        "target_ids": list(missing),
        "context_ids": [span_id for span_id in dict.fromkeys(context) if span_id not in missing],
    }


def _messages(prompt, spans, window):
    payload = {
        "target_spans": {span_id: spans[span_id]["text"] for span_id in window["target_ids"]},
        "context_spans": {span_id: spans[span_id]["text"] for span_id in window["context_ids"]},
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _call_window(llm, prompt, spans, window, first_event_number):
    attempts = []
    changes = []
    decoded = None
    missing = list(window["target_ids"])
    base_messages = _messages(prompt, spans, window)
    original_messages = base_messages
    for number in range(1, 3):
        raw = ""
        try:
            llm.last_generation = {}
            raw = llm._generate(original_messages, max_new_tokens=1536)
            decoded, missing, normalized = _decode_partial(json.loads(raw), spans, window, first_event_number)
            changes.extend(normalized)
            attempts.append({"attempt": number, "status": "ok", "adopted": True, "raw_output": raw,
                             "coverage_pending": list(missing), "timing": dict(llm.last_generation)})
            break
        except (ValueError, RuntimeError, OSError) as exc:
            attempts.append({"attempt": number, "status": "error", "raw_output": getattr(exc, "raw_output", raw),
                             "error": str(exc), "error_type": type(exc).__name__,
                             "timing": dict(getattr(llm, "last_generation", {}))})
            if number == 1 and isinstance(exc, ValueError):
                original_messages = original_messages + [{
                    "role": "user", "content": "上次回复不符合协议：" + str(exc) + "。请重新输出本窗口的有效JSON。"
                }]
                continue
            return None, {"status": "error", "attempts": attempts, "wire_normalizations": list(dict.fromkeys(changes)),
                          "uncovered_target_ids": list(window["target_ids"])}

    recovered_target_ids = []
    for _ in range(2):
        if not missing:
            break
        local_window = _recovery_window(spans, window, missing)
        raw = ""
        try:
            llm.last_generation = {}
            raw = llm._generate(_messages(prompt, spans, local_window), max_new_tokens=1536)
            recovered, remaining, normalized = _decode_partial(
                json.loads(raw), spans, local_window, first_event_number + len(decoded[0])
            )
            changes.extend(normalized)
            recovered_target_ids.extend(span_id for span_id in missing if span_id not in remaining)
            decoded = tuple(left + right for left, right in zip(decoded, recovered))
            attempts.append({"attempt": len(attempts) + 1, "purpose": "recover_missing_targets",
                             "target_ids": list(missing), "status": "ok", "adopted": True, "raw_output": raw,
                             "coverage_pending": list(remaining), "timing": dict(llm.last_generation)})
            missing = remaining
        except (ValueError, RuntimeError, OSError) as exc:
            attempts.append({"attempt": len(attempts) + 1, "purpose": "recover_missing_targets",
                             "target_ids": list(missing), "status": "error", "raw_output": getattr(exc, "raw_output", raw),
                             "error": str(exc), "error_type": type(exc).__name__,
                             "timing": dict(getattr(llm, "last_generation", {}))})
            break
    if missing:
        raw = ""
        try:
            llm.last_generation = {}
            raw = llm._generate(base_messages + [{
                "role": "user",
                "content": "上次回复不合格：存在未覆盖的target_spans：" + ",".join(missing)
                           + "。请重新审计本窗口的全部target_spans，只返回符合协议的完整JSON对象。",
            }], max_new_tokens=1536)
            replacement, replacement_missing, normalized = _decode_partial(
                json.loads(raw), spans, window, first_event_number
            )
            changes.extend(normalized)
            adopted = not replacement_missing
            attempts.append({"attempt": len(attempts) + 1, "purpose": "fallback_full_window",
                             "status": "ok", "adopted": adopted, "raw_output": raw,
                             "coverage_pending": list(replacement_missing), "timing": dict(llm.last_generation)})
            if adopted:
                for prior in attempts[:-1]:
                    if prior["status"] == "ok":
                        prior["adopted"] = False
                decoded, missing = replacement, []
        except (ValueError, RuntimeError, OSError) as exc:
            attempts.append({"attempt": len(attempts) + 1, "purpose": "fallback_full_window",
                             "status": "error", "raw_output": getattr(exc, "raw_output", raw),
                             "error": str(exc), "error_type": type(exc).__name__,
                             "timing": dict(getattr(llm, "last_generation", {}))})
    return decoded, {
        "status": "ok" if not missing else "partial", "attempts": attempts,
        "wire_normalizations": list(dict.fromkeys(changes)), "recovered_target_ids": recovered_target_ids,
        "uncovered_target_ids": list(missing),
    }


def extract_events_b(text: str, device: str = "auto", llm=None, progress=None) -> dict:
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
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    events, ignored, non_events, calls, rejected, uncovered = [], [], [], [], [], []
    for window_id, window in enumerate(_windows(spans), 1):
        if progress:
            progress({"window_id": window_id, "purpose": "extract_checkable_events", "target_ids": window["target_ids"]})
        decoded, call = _call_window(llm, prompt, spans, window, len(events) + 1)
        call.update({"window_id": window_id, "purpose": "extract_checkable_events", **window})
        calls.append(call)
        if decoded is not None:
            window_events, window_ignored, window_non_events = decoded
            events.extend(window_events)
            ignored.extend(window_ignored)
            non_events.extend(window_non_events)
        if call["uncovered_target_ids"]:
            pending = call["uncovered_target_ids"]
            uncovered.extend(pending)
            rejected.append({"window_id": window_id, "target_ids": pending,
                             "error": "恢复后仍有未覆盖的target_spans：" + ",".join(pending)})
    status = "ok" if not uncovered else "error" if len(uncovered) == len(spans) else "partial"
    accounting = account_coverage(spans, events, ignored, non_events)
    if set(accounting["uncovered_span_ids"]) != set(uncovered):
        raise ValueError("程序覆盖账本与提取报告不一致")
    uncovered = sorted(uncovered)
    result = {
        "schema_version": SCHEMA_VERSION, "prompt_version": "extractor-v1+deterministic-coverage-b",
        "stage": "extraction", "status": status, "text": text, "spans": spans,
        "events": events, "ignored_spans": ignored, "non_event_span_ids": non_events,
        "uncovered_span_ids": uncovered, "coverage_accounting": accounting,
        "rejected": rejected, "calls": calls, "processing_complete": not uncovered,
        "model_loaded_this_request": loaded_here, "device": getattr(llm, "device", device),
        "model_load_seconds": getattr(llm, "load_seconds", None),
        "request_seconds": time.perf_counter() - started,
    }
    validate_extraction(result)
    return result
