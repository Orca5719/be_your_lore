from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CATEGORIES = (
    "JSON_PARSE_ERROR",
    "TRUNCATED_OUTPUT",
    "TOP_LEVEL_SCHEMA_ERROR",
    "EVENT_FIELD_ERROR",
    "INVALID_REFERENCE_OR_DISPOSITION",
    "INVALID_ENUM_OR_TYPE",
    "MISSING_TARGET_COVERAGE",
    "EMPTY_OUTPUT",
    "OTHER",
)
COST_CLASSES = ("pure_waste", "recovery_cost", "effective_cost")
CATEGORY_DETAILS = {
    "JSON_PARSE_ERROR": ("JSON transport", "输出不是可解析的完整JSON对象"),
    "TRUNCATED_OUTPUT": ("output budget", "生成到达上限，结构未完整结束"),
    "TOP_LEVEL_SCHEMA_ERROR": ("top-level keys", "顶层字段集合不符合Extractor wire协议"),
    "EVENT_FIELD_ERROR": ("events[]", "事件对象缺失、增加或错误使用必需字段"),
    "INVALID_REFERENCE_OR_DISPOSITION": ("source/context and disposition IDs", "span引用越界、重复或处置对象结构无效"),
    "INVALID_ENUM_OR_TYPE": ("check_reason/modality/reason and field types", "枚举值或字段类型不在协议允许范围"),
    "MISSING_TARGET_COVERAGE": ("coverage ledger", "至少一个target span未进入event、ignored或non-event任一路径"),
    "EMPTY_OUTPUT": ("transport", "模型没有返回可检查内容"),
    "OTHER": ("unknown", "错误不匹配已知协议失败模式，需要逐案复核"),
}


def classify_failure(*, error: str | None, error_type: str | None, raw_output: str | None) -> str:
    error = str(error or "")
    error_type = str(error_type or "")
    raw = str(raw_output or "")
    lowered = error.lower()
    if "未覆盖的target_spans" in error or "missing target" in lowered:
        return "MISSING_TARGET_COVERAGE"
    if not raw.strip():
        return "EMPTY_OUTPUT"
    if "jsondecode" in error_type.lower() or "expecting " in lowered or "有效 json" in lowered:
        return "JSON_PARSE_ERROR"
    if "输出上限" in error or "truncat" in lowered or "max_new_tokens" in lowered:
        return "TRUNCATED_OUTPUT"
    if "输出必须且只能包含" in error or "Lean wire必须且只能包含" in error or "顶层" in error and "字段" in error:
        return "TOP_LEVEL_SCHEMA_ERROR"
    if "事件字段" in error or "event字段" in lowered:
        return "EVENT_FIELD_ERROR"
    reference_markers = ("只能引用", "引用无效", "超出当前窗口", "不能重叠", "ignored_spans字段无效", "处置")
    if any(marker in error for marker in reference_markers):
        return "INVALID_REFERENCE_OR_DISPOSITION"
    enum_markers = ("核对原因无效", "modality", "reason无效", "必须为", "须为")
    if any(marker in error for marker in enum_markers):
        return "INVALID_ENUM_OR_TYPE"
    return "OTHER"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"missing required file: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _call_map(calls: list[dict]) -> dict[str, dict]:
    result = {}
    for call in calls:
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in result:
            raise ValueError("invalid or duplicate inference call_id")
        result[call_id] = call
    return result


def build_audit_rows(stories: list[dict], calls: list[dict]) -> list[dict]:
    extractor_calls = [call for call in calls if call.get("component") == "extractor"]
    by_id = _call_map(extractor_calls)
    metadata = {}
    for story in stories:
        case_id = story.get("case_id")
        extraction = story.get("result", {}).get("benchmark_stages", {}).get("extraction", {})
        for window in extraction.get("calls", []):
            attempts = window.get("attempts", [])
            for position, attempt in enumerate(attempts):
                call_id = attempt.get("timing", {}).get("call_id")
                if not call_id:
                    continue
                if call_id in metadata:
                    raise ValueError(f"duplicate extractor attempt mapping: {call_id}")
                later = attempts[position + 1:]
                previous_failed = any(value.get("status") == "error" for value in attempts[:position])
                status = attempt.get("status")
                purpose = attempt.get("purpose") or window.get("purpose") or "extract_checkable_events"
                is_recovery = purpose in {"recover_missing_targets", "recover_disposition"}
                if status == "error":
                    cost_class = "pure_waste"
                    outcome = "recovery_failure" if is_recovery else "format_retry_failure" if previous_failed else "initial_failure"
                elif is_recovery:
                    cost_class = "recovery_cost"
                    outcome = "recovery_success"
                else:
                    cost_class = "effective_cost"
                    outcome = "format_retry_success" if previous_failed else "initial_success"
                raw = str(attempt.get("raw_output") or "")
                failure_category = None if status == "ok" else classify_failure(
                    error=attempt.get("error"), error_type=attempt.get("error_type"), raw_output=raw
                )
                metadata[call_id] = {
                    "case_id": case_id,
                    "window_id": window.get("window_id"),
                    "target_ids": list(window.get("target_ids", [])),
                    "attempt": attempt.get("attempt", position + 1),
                    "attempt_position": position + 1,
                    "purpose": purpose,
                    "attempt_status": status,
                    "error_type": attempt.get("error_type"),
                    "error": attempt.get("error"),
                    "failure_category": failure_category,
                    "cost_class": cost_class,
                    "outcome": outcome,
                    "recovered_by_later": status == "error" and any(value.get("status") == "ok" for value in later),
                    "final_window_status": "ok" if any(value.get("status") == "ok" for value in attempts) else "error",
                    "raw_output_length": len(raw),
                    "raw_output_summary": raw[:300],
                }
    if set(metadata) != set(by_id):
        missing = sorted(set(by_id) - set(metadata))
        extra = sorted(set(metadata) - set(by_id))
        raise ValueError(f"extractor attempt/call mapping mismatch: missing={missing}, extra={extra}")
    return [{**by_id[call_id], **metadata[call_id]} for call_id in by_id]


def _bucket(rows: list[dict], total_seconds: float) -> dict:
    stories = sorted({str(row.get("case_id")) for row in rows})
    windows = sorted({f"{row.get('case_id')}:{row.get('window_id')}" for row in rows})
    return {
        "calls": len(rows),
        "stories": len(stories),
        "windows": len(windows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "prefill_time": sum(float(row.get("prefill_time") or 0.0) for row in rows),
        "decode_time": sum(float(row.get("decode_time") or 0.0) for row in rows),
        "total_time": sum(float(row.get("total_time") or 0.0) for row in rows),
        "time_share": sum(float(row.get("total_time") or 0.0) for row in rows) / total_seconds if total_seconds else None,
        "recovered_calls": sum(bool(row.get("recovered_by_later")) for row in rows),
        "case_ids": stories,
    }


def aggregate_audit(rows: list[dict]) -> dict:
    if any(row.get("cost_class") not in COST_CLASSES for row in rows):
        raise ValueError("every extractor call must have exactly one cost class")
    total_seconds = sum(float(row.get("total_time") or 0.0) for row in rows)
    totals = _bucket(rows, total_seconds)
    cost_classes = {name: _bucket([row for row in rows if row["cost_class"] == name], total_seconds) for name in COST_CLASSES}
    failures = [row for row in rows if row.get("failure_category")]
    categories = {}
    for category in CATEGORIES:
        selected = [row for row in failures if row["failure_category"] == category]
        item = _bucket(selected, total_seconds)
        field, cause = CATEGORY_DETAILS[category]
        item.update({
            "related_schema": field,
            "direct_cause": cause,
            "examples": [
                {"call_id": row["call_id"], "case_id": row["case_id"], "window_id": row["window_id"],
                 "error": row.get("error"), "raw_output_summary": row.get("raw_output_summary", "")}
                for row in selected[:5]
            ],
        })
        categories[category] = item
    outcomes = {name: _bucket([row for row in rows if row["outcome"] == name], total_seconds) for name in (
        "initial_success", "initial_failure", "format_retry_success", "format_retry_failure", "recovery_success", "recovery_failure"
    )}
    sum_calls = sum(value["calls"] for value in cost_classes.values())
    sum_input = sum(value["input_tokens"] for value in cost_classes.values())
    sum_output = sum(value["output_tokens"] for value in cost_classes.values())
    sum_prefill = sum(value["prefill_time"] for value in cost_classes.values())
    sum_decode = sum(value["decode_time"] for value in cost_classes.values())
    sum_time = sum(value["total_time"] for value in cost_classes.values())
    closure = {
        "calls_closed": sum_calls == totals["calls"],
        "tokens_closed": sum_input == totals["input_tokens"] and sum_output == totals["output_tokens"],
        "time_closed": all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in (
            (sum_prefill, totals["prefill_time"]), (sum_decode, totals["decode_time"]), (sum_time, totals["total_time"])
        )),
    }
    if not all(closure.values()):
        raise ValueError("extractor cost accounting does not close")
    recommendations = [
        {"rank": index + 1, "category": category, "wasted_seconds": item["total_time"],
         "wasted_output_tokens": item["output_tokens"], "related_schema": item["related_schema"],
         "hypothesis": item["direct_cause"]}
        for index, (category, item) in enumerate(sorted(
            ((key, value) for key, value in categories.items() if value["calls"]),
            key=lambda pair: pair[1]["total_time"], reverse=True,
        ))
    ]
    return {
        "schema_version": "agent-pipeline-v3.2-extractor-failure-audit-v1",
        "totals": totals,
        "cost_classes": cost_classes,
        "failure_categories": categories,
        "outcomes": outcomes,
        "closure": closure,
        "recommendations": recommendations,
    }


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_extractor_audit(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("audit output must be outside the frozen source directory")
    story_path = source / "story_runs.jsonl"
    call_path = source / "inference_calls.jsonl"
    if not story_path.exists() or not call_path.exists():
        raise ValueError("Benchmark 3 story_runs.jsonl and inference_calls.jsonl are required")
    stories = _read_jsonl(story_path)
    calls = _read_jsonl(call_path)
    if len(stories) != 24:
        raise ValueError("3.2B requires the fixed 24-story Benchmark 3 result")
    rows = build_audit_rows(stories, calls)
    if len(rows) != 103:
        raise ValueError(f"3.2B requires 103 extractor calls; found {len(rows)}")
    report = aggregate_audit(rows)
    report.update({
        "source": str(source),
        "source_hashes": {"story_runs.jsonl": _sha(story_path), "inference_calls.jsonl": _sha(call_path)},
        "stories": 24,
        "extractor_calls": len(rows),
        "model_loaded": False,
        "generate_called": False,
    })
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(output / "extractor_calls_audit.jsonl", rows)
    (output / "extractor_failure_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "extractor_failure_categories.csv", [
        {"category": category, **value} for category, value in report["failure_categories"].items()
    ], ["category", "calls", "stories", "windows", "input_tokens", "output_tokens", "prefill_time", "decode_time", "total_time", "time_share", "recovered_calls", "related_schema", "direct_cause"])
    _write_csv(output / "extractor_calls_audit.csv", rows, [
        "call_id", "case_id", "window_id", "attempt", "attempt_position", "purpose", "attempt_status", "outcome", "cost_class", "failure_category", "recovered_by_later", "input_tokens", "output_tokens", "prefill_time", "decode_time", "total_time", "error_type", "error", "raw_output_length", "raw_output_summary"
    ])
    return report
