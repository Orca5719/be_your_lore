from __future__ import annotations

from agent_pipeline_v2_2.schema import stable_digest


REFERENCE_SCHEMA = "agent-pipeline-v3-quality-reference-v2"


def _without(value: dict, excluded: set[str]) -> dict:
    return {key: item for key, item in value.items() if key not in excluded}


def _semantic_evidence(value: dict) -> dict:
    """Keep evidence identity/content/order while excluding numeric retrieval diagnostics."""
    fields = ("id", "text", "file", "start_line", "end_line", "heading_path", "rank")
    return {field: value.get(field) for field in fields if field in value}


def semantic_payload(row: dict) -> dict:
    result = row.get("result", {})
    stages = result.get("benchmark_stages", {})
    extraction = stages.get("extraction", {})
    judge = stages.get("judge", {})
    events = [_without(event, {"review_required", "review_reasons"}) for event in extraction.get("events", [])]
    items = []
    for item in judge.get("items", []):
        normalized = _without(item, {"call", "evidence"})
        normalized["evidence"] = [_semantic_evidence(value) for value in item.get("evidence", [])]
        items.append(normalized)
    findings = list(result.get("findings", []))
    return {
        "case_id": row.get("case_id"),
        "status": row.get("status"),
        "extraction_status": extraction.get("status"),
        "events": events,
        "judge_status": judge.get("status"),
        "judge_items": items,
        "report_summary": result.get("summary"),
        "findings": findings,
    }


def build_quality_reference(rows: list[dict], official_metrics: dict | None = None) -> dict:
    cases = {row["case_id"]: stable_digest(semantic_payload(row)) for row in rows}
    if len(cases) != len(rows):
        raise ValueError("duplicate quality reference case IDs")
    return {
        "schema_version": REFERENCE_SCHEMA,
        "cases": dict(sorted(cases.items())),
        "official_metrics": official_metrics or {},
    }


def compare_quality(rows: list[dict], reference: dict) -> dict:
    if reference.get("schema_version") != REFERENCE_SCHEMA or not isinstance(reference.get("cases"), dict):
        raise ValueError("invalid quality reference")
    actual = {row["case_id"]: stable_digest(semantic_payload(row)) for row in rows}
    expected = reference["cases"]
    shared = sorted(set(actual) & set(expected))
    mismatched = [case_id for case_id in shared if actual[case_id] != expected[case_id]]
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    comparable = not mismatched and not missing and not unexpected
    return {
        "comparable": comparable,
        "matched_cases": sum(actual[case_id] == expected[case_id] for case_id in shared),
        "total_reference_cases": len(expected),
        "mismatched_case_ids": mismatched,
        "missing_case_ids": missing,
        "unexpected_case_ids": unexpected,
        "official_metrics": reference.get("official_metrics", {}),
    }
