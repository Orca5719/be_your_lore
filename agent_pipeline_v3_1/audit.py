from __future__ import annotations

import csv
import json
from pathlib import Path

from .taxonomy import RETRY_CATEGORIES, classify_failure, classify_retry


AUDIT_SCHEMA = "agent-pipeline-v3.1-retry-audit-v1"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"missing required file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"missing required file: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _by_unique_call_id(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in result:
            raise ValueError(f"duplicate or invalid call_id in {label}")
        result[call_id] = row
    return result


def validate_result(
    directory: Path,
    *,
    expected_stories: int = 24,
    expected_calls: int = 130,
    expected_extractor: int = 103,
    expected_judge: int = 27,
) -> dict:
    directory = Path(directory)
    stories = _read_jsonl(directory / "story_runs.jsonl")
    flat_calls = _read_jsonl(directory / "inference_calls.jsonl")
    profile = _read_json(directory / "profile_summary.json")
    case_ids = [row.get("case_id") for row in stories]
    if len(stories) != expected_stories or len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise ValueError("story count or IDs do not match the fixed audit input")
    quality = profile.get("quality_guard", {})
    if not quality.get("comparable") or quality.get("matched_cases") != expected_stories or quality.get("total_reference_cases") != expected_stories:
        raise ValueError("quality guard must pass for every story")
    nested_calls = [call for story in stories for call in story.get("calls", [])]
    nested = _by_unique_call_id(nested_calls, "story_runs.jsonl")
    flat = _by_unique_call_id(flat_calls, "inference_calls.jsonl")
    if len(flat) != expected_calls:
        raise ValueError("LLM call count does not match the fixed audit input")
    if nested != flat:
        raise ValueError("call records do not reconcile")
    counts = {component: sum(row.get("component") == component for row in flat_calls) for component in ("extractor", "judge")}
    if counts != {"extractor": expected_extractor, "judge": expected_judge}:
        raise ValueError("component call counts do not match the fixed audit input")
    attempt_ids = set(_attempt_metadata(stories))
    call_ids = set(flat)
    if attempt_ids != call_ids:
        missing = sorted(call_ids - attempt_ids)
        extra = sorted(attempt_ids - call_ids)
        raise ValueError(f"attempt/call mapping incomplete: missing={missing}, extra={extra}")
    return {
        "status": "ok",
        "stories": len(stories),
        "calls": len(flat_calls),
        "extractor_calls": counts["extractor"],
        "judge_calls": counts["judge"],
        "quality_matched": quality["matched_cases"],
    }


def _attempt_metadata(stories: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for story in stories:
        case_id = story["case_id"]
        stages = story.get("result", {}).get("benchmark_stages", {})
        extraction = stages.get("extraction", {})
        for window in extraction.get("calls", []):
            attempts = window.get("attempts", [])
            for index, attempt in enumerate(attempts):
                call_id = attempt.get("timing", {}).get("call_id")
                if not call_id:
                    continue
                previous = attempts[index - 1] if index else None
                metadata[call_id] = {
                    "case_id": case_id,
                    "component": "extractor",
                    "attempt": int(attempt.get("attempt", index + 1)),
                    "window_id": window.get("window_id"),
                    "purpose": attempt.get("purpose") or window.get("purpose"),
                    "attempt_status": attempt.get("status"),
                    "attempt_error": attempt.get("error"),
                    "attempt_error_type": attempt.get("error_type"),
                    "previous": previous,
                }
        judge = stages.get("judge", {})
        for report in judge.get("batch_reports", []):
            row_attempts = {row.get("request_id"): row.get("attempt_records", []) for row in report.get("rows", [])}
            for batch_call in report.get("batch_calls", []):
                call_id = batch_call.get("timing", {}).get("call_id")
                if not call_id:
                    continue
                attempt_number = int(batch_call.get("attempt", 1))
                request_ids = list(batch_call.get("request_ids", []))
                previous_rows = []
                for request_id in request_ids:
                    attempts = row_attempts.get(request_id, [])
                    if attempt_number > 1 and len(attempts) >= attempt_number - 1:
                        previous_rows.append(attempts[attempt_number - 2])
                metadata[call_id] = {
                    "case_id": case_id,
                    "component": "judge",
                    "attempt": attempt_number,
                    "window_id": None,
                    "purpose": "judge_row_retry" if attempt_number > 1 else "batched_judge",
                    "attempt_status": batch_call.get("status"),
                    "attempt_error": batch_call.get("error"),
                    "attempt_error_type": batch_call.get("error_type"),
                    "request_ids": request_ids,
                    "previous_rows": previous_rows,
                }
    return metadata


def _audit_rows(stories: list[dict], calls: list[dict]) -> list[dict]:
    metadata = _attempt_metadata(stories)
    call_ids = {row["call_id"] for row in calls}
    if set(metadata) != call_ids:
        missing = sorted(call_ids - set(metadata))
        extra = sorted(set(metadata) - call_ids)
        raise ValueError(f"attempt/call mapping incomplete: missing={missing}, extra={extra}")
    rows = []
    for call in calls:
        meta = metadata[call["call_id"]]
        retry = bool(call.get("is_retry"))
        category = None
        retry_kind = None
        trigger_errors = []
        underlying_categories = []
        if retry:
            if call["component"] == "extractor":
                previous = meta.get("previous") or {}
                category = classify_retry("extractor", {"purpose": meta.get("purpose"), "raw_output": "present"}, previous)
                retry_kind = "coverage_recovery" if category == "MISSING_TARGET_COVERAGE" else "failed_attempt"
                if previous.get("error"):
                    trigger_errors.append({"error_type": previous.get("error_type"), "error": previous.get("error")})
            else:
                previous_rows = meta.get("previous_rows", [])
                category = classify_retry("judge", {}, previous_rows[0] if previous_rows else {})
                retry_kind = "judge_row_retry"
                trigger_errors = [
                    {"error_type": row.get("error_type"), "error": row.get("error")}
                    for row in previous_rows if row.get("error")
                ]
                underlying_categories = [classify_failure({}, row) for row in previous_rows]
        rows.append({
            **call,
            "case_id": meta["case_id"],
            "attempt": meta["attempt"],
            "window_id": meta.get("window_id"),
            "purpose": meta.get("purpose"),
            "attempt_status": meta.get("attempt_status"),
            "attempt_error": meta.get("attempt_error"),
            "attempt_error_type": meta.get("attempt_error_type"),
            "retry_category": category,
            "retry_kind": retry_kind,
            "trigger_errors": trigger_errors,
            "underlying_categories": underlying_categories,
        })
    return rows


def _bucket(rows: list[dict], key: str, values: tuple[str, ...] | list[str]) -> dict:
    result = {}
    for value in values:
        selected = [row for row in rows if row.get(key) == value]
        result[value] = {
            "calls": len(selected),
            "input_tokens": sum(row["input_tokens"] for row in selected),
            "output_tokens": sum(row["output_tokens"] for row in selected),
            "prefill_time": sum((row.get("prefill_time") or 0.0) for row in selected),
            "decode_time": sum((row.get("decode_time") or 0.0) for row in selected),
            "total_time": sum(row["total_time"] for row in selected),
            "case_ids": sorted({row["case_id"] for row in selected}),
        }
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def audit_retries(
    source: Path,
    output: Path,
    *,
    expected_stories: int = 24,
    expected_calls: int = 130,
    expected_extractor: int = 103,
    expected_judge: int = 27,
) -> dict:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("audit output must be outside the frozen source directory")
    validation = validate_result(
        source,
        expected_stories=expected_stories,
        expected_calls=expected_calls,
        expected_extractor=expected_extractor,
        expected_judge=expected_judge,
    )
    stories = _read_jsonl(source / "story_runs.jsonl")
    calls = _read_jsonl(source / "inference_calls.jsonl")
    audited = _audit_rows(stories, calls)
    retry_rows = [row for row in audited if row.get("is_retry")]
    if len(retry_rows) != 31 and expected_calls == 130:
        raise ValueError("fixed Benchmark 3 input must contain 31 retry calls")
    judge_causes = []
    for row in retry_rows:
        for category in row.get("underlying_categories", []):
            judge_causes.append({**row, "underlying_category": category})
    report = {
        "schema_version": AUDIT_SCHEMA,
        "source": str(source),
        "validation": validation,
        "retry_calls": len(retry_rows),
        "categories": _bucket(retry_rows, "retry_category", RETRY_CATEGORIES),
        "retry_kinds": _bucket(retry_rows, "retry_kind", ["failed_attempt", "coverage_recovery", "judge_row_retry"]),
        "judge_retry_causes": _bucket(judge_causes, "underlying_category", RETRY_CATEGORIES),
        "totals": {
            "input_tokens": sum(row["input_tokens"] for row in retry_rows),
            "output_tokens": sum(row["output_tokens"] for row in retry_rows),
            "prefill_time": sum((row.get("prefill_time") or 0.0) for row in retry_rows),
            "decode_time": sum((row.get("decode_time") or 0.0) for row in retry_rows),
            "total_time": sum(row["total_time"] for row in retry_rows),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "calls_audit.jsonl", audited)
    (output / "retry_taxonomy.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["call_id", "case_id", "component", "attempt", "purpose", "retry_category", "retry_kind", "input_tokens", "output_tokens", "prefill_time", "decode_time", "total_time", "trigger_errors"]
    with (output / "retry_taxonomy.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in retry_rows:
            writer.writerow({**row, "trigger_errors": json.dumps(row["trigger_errors"], ensure_ascii=False)})
    return report
