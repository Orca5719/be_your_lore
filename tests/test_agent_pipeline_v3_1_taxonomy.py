import pytest

from agent_pipeline_v3_1.taxonomy import classify_retry


@pytest.mark.parametrize(
    ("component", "attempt", "previous", "expected"),
    [
        ("extractor", {"raw_output": "{"}, {"error_type": "JSONDecodeError", "error": "bad json"}, "JSON_PARSE_ERROR"),
        ("extractor", {"raw_output": "{}"}, {"error_type": "ValueError", "error": "ignored_spans字段无效"}, "SCHEMA_VALIDATION_ERROR"),
        ("extractor", {"purpose": "recover_missing_targets"}, {"error": "存在未覆盖的target_spans：S1"}, "MISSING_TARGET_COVERAGE"),
        ("extractor", {"raw_output": ""}, {"error": "empty"}, "EMPTY_OUTPUT"),
        ("extractor", {"raw_output": "{}"}, {"error": "生成达到输出上限且未结束"}, "TRUNCATED_OUTPUT"),
        ("extractor", {"raw_output": "{}"}, {"error": "non_event_span_ids只能引用当前target_spans"}, "INVALID_FIELD_OR_REFERENCE"),
        ("judge", {"raw_output": "{}"}, {"error": "citation字段无效"}, "BATCH_ROW_RETRY"),
        ("extractor", {"raw_output": "{}"}, {"error": "unclassified"}, "OTHER"),
    ],
)
def test_classify_retry_covers_fixed_taxonomy(component, attempt, previous, expected):
    assert classify_retry(component, attempt, previous) == expected
