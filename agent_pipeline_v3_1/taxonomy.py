from __future__ import annotations


RETRY_CATEGORIES = (
    "JSON_PARSE_ERROR",
    "SCHEMA_VALIDATION_ERROR",
    "MISSING_TARGET_COVERAGE",
    "EMPTY_OUTPUT",
    "TRUNCATED_OUTPUT",
    "INVALID_FIELD_OR_REFERENCE",
    "BATCH_ROW_RETRY",
    "OTHER",
)


def classify_failure(attempt: dict, previous: dict | None) -> str:
    """Classify the validation/generation failure that triggered another call."""
    previous = previous or {}
    purpose = str(attempt.get("purpose", ""))
    error_type = str(previous.get("error_type", ""))
    error = str(previous.get("error", ""))
    raw = previous.get("raw_output", attempt.get("raw_output", ""))
    lowered = error.lower()
    if purpose == "recover_missing_targets" or "未覆盖的target_spans" in error or "missing target" in lowered:
        return "MISSING_TARGET_COVERAGE"
    if not str(raw or "").strip():
        return "EMPTY_OUTPUT"
    if "jsondecode" in error_type.lower() or "有效 json" in lowered or "valid json" in lowered:
        return "JSON_PARSE_ERROR"
    if "输出上限" in error or "truncat" in lowered or "max_new_tokens" in lowered:
        return "TRUNCATED_OUTPUT"
    reference_markers = ("只能引用", "引用无效", "citation字段无效", "source_id无效", "context_id无效")
    if any(marker in error for marker in reference_markers):
        return "INVALID_FIELD_OR_REFERENCE"
    schema_markers = ("字段无效", "字段不符合", "wire协议", "输出必须", "核对原因无效", "schema")
    if any(marker in error for marker in schema_markers):
        return "SCHEMA_VALIDATION_ERROR"
    return "OTHER"


def classify_retry(component: str, attempt: dict, previous: dict | None) -> str:
    """Classify a retry call while retaining Judge batching as the primary category."""
    if component == "judge":
        return "BATCH_ROW_RETRY"
    return classify_failure(attempt, previous)
