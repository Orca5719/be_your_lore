from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable


TokenCounter = Callable[[str], int]


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tokens(counter: TokenCounter, value: object) -> int:
    return int(counter(_compact(value)))


def project_extractor_payload(payload: dict) -> dict:
    """Keep downstream event data and the minimum coverage ledger."""
    ignored_ids = []
    for item in payload.get("ignored_spans", []):
        if isinstance(item, dict) and item.get("source_id"):
            ignored_ids.append(item["source_id"])
        elif isinstance(item, str):
            ignored_ids.append(item)
    return {
        "events": payload.get("events", []),
        "ignored_span_ids": ignored_ids,
        "non_event_span_ids": payload.get("non_event_span_ids", []),
    }


def project_judge_payload(payload: dict) -> dict:
    """Keep every verdict; conflicts retain evidence identity and a short reason."""
    verdict = payload.get("verdict")
    projected = {"verdict": verdict}
    if verdict == "contradiction":
        projected["citation_ids"] = [
            item.get("evidence_id") or item.get("chunk_id")
            for item in payload.get("citations", [])
            if isinstance(item, dict) and (item.get("evidence_id") or item.get("chunk_id"))
        ]
        projected["reason"] = str(payload.get("reason") or "")[:80]
    return projected


def _extractor_attempts(stories: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for story in stories:
        extraction = story.get("result", {}).get("benchmark_stages", {}).get("extraction", {})
        for window in extraction.get("calls", []):
            for attempt in window.get("attempts", []):
                call_id = attempt.get("timing", {}).get("call_id")
                if call_id:
                    result[call_id] = attempt.get("raw_output") or ""
    return result


def _judge_attempts(stories: list[dict]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for story in stories:
        judge = story.get("result", {}).get("benchmark_stages", {}).get("judge", {})
        for report in judge.get("batch_reports", []):
            rows = {row.get("request_id"): row for row in report.get("rows", [])}
            for batch_call in report.get("batch_calls", []):
                call_id = batch_call.get("timing", {}).get("call_id")
                if not call_id:
                    continue
                attempt_number = int(batch_call.get("attempt", 1))
                outputs = []
                for request_id in batch_call.get("request_ids", []):
                    attempts = rows.get(request_id, {}).get("attempt_records", [])
                    if len(attempts) >= attempt_number:
                        outputs.append(attempts[attempt_number - 1].get("raw_output") or "")
                result[call_id] = outputs
    return result


def _component_template() -> dict:
    return {
        "calls": 0,
        "actual_output_tokens": 0,
        "actual_decode_time": 0.0,
        "field_estimated_tokens": defaultdict(int),
        "transport_residual_tokens": 0,
        "projected_output_tokens": 0,
        "unprojectable_output_tokens": 0,
        "parse_failures": 0,
        "output_classes": defaultdict(lambda: {"calls": 0, "actual_output_tokens": 0}),
        "verdicts": defaultdict(lambda: {"rows": 0, "estimated_tokens": 0}),
        "projection_kind": "counterfactual_estimate",
    }


def _field_values(component: str, payload: dict) -> dict[str, object]:
    if component == "extractor":
        return {
            "events": payload.get("events", []),
            "ignored_spans": payload.get("ignored_spans", []),
            "non_event_spans": payload.get("non_event_spans", payload.get("non_event_span_ids", [])),
        }
    return {
        "verdict": payload.get("verdict"),
        "citations": payload.get("citations", []),
        "reason": payload.get("reason"),
        "assessment": payload.get("assessment", {}),
    }


def analyze_generation(stories: list[dict], calls: list[dict], token_counter: TokenCounter) -> dict:
    """Analyze saved outputs only. This function never loads or calls a model."""
    raw_by_component = {
        "extractor": _extractor_attempts(stories),
        "judge": _judge_attempts(stories),
    }
    components = {"extractor": _component_template(), "judge": _component_template()}
    call_rows = []
    field_rows = []
    projection_rows = []

    for call in calls:
        component = call.get("component")
        if component not in components:
            continue
        aggregate = components[component]
        actual = int(call.get("output_tokens") or 0)
        aggregate["calls"] += 1
        aggregate["actual_output_tokens"] += actual
        aggregate["actual_decode_time"] += float(call.get("decode_time") or 0.0)
        raws = raw_by_component[component].get(call.get("call_id"), [])
        if isinstance(raws, str):
            raws = [raws]
        parsed: list[dict] = []
        for raw in raws:
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("output is not an object")
                parsed.append(payload)
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = []
                break

        estimated_fields: dict[str, int] = defaultdict(int)
        if not parsed:
            aggregate["parse_failures"] += 1
            aggregate["unprojectable_output_tokens"] += actual
            aggregate["projected_output_tokens"] += actual
            projected = actual
            output_class = "failed_output"
        else:
            projected = 0
            for payload in parsed:
                for field, value in _field_values(component, payload).items():
                    count = _tokens(token_counter, value)
                    estimated_fields[field] += count
                    aggregate["field_estimated_tokens"][field] += count
                full_estimate = _tokens(token_counter, payload)
                structure_estimate = max(0, full_estimate - sum(
                    _tokens(token_counter, value) for value in _field_values(component, payload).values()
                ))
                estimated_fields["json_structure"] += structure_estimate
                aggregate["field_estimated_tokens"]["json_structure"] += structure_estimate
                projection = project_extractor_payload(payload) if component == "extractor" else project_judge_payload(payload)
                projected += _tokens(token_counter, projection)
                if component == "judge":
                    verdict = str(payload.get("verdict") or "unknown")
                    aggregate["verdicts"][verdict]["rows"] += 1
                    aggregate["verdicts"][verdict]["estimated_tokens"] += sum(
                        _tokens(token_counter, value) for value in _field_values(component, payload).values()
                    )
            aggregate["projected_output_tokens"] += projected
            aggregate["transport_residual_tokens"] += actual - sum(estimated_fields.values())
            if call.get("retry_kind") == "coverage_recovery" or call.get("purpose") == "coverage_recovery":
                output_class = "coverage_recovery"
            elif component == "extractor" and call.get("attempt_status") not in (None, "ok"):
                output_class = "failed_output"
            else:
                output_class = "successful_content"

        aggregate["output_classes"][output_class]["calls"] += 1
        aggregate["output_classes"][output_class]["actual_output_tokens"] += actual

        for field, count in estimated_fields.items():
            field_rows.append({"call_id": call.get("call_id"), "component": component, "field": field, "estimated_tokens": count})
        projection_rows.append({
            "call_id": call.get("call_id"), "component": component,
            "actual_output_tokens": actual, "projected_output_tokens": projected,
            "projection_kind": "counterfactual_estimate",
        })
        call_rows.append({**call, "projected_output_tokens": projected})

    for aggregate in components.values():
        aggregate["field_estimated_tokens"] = dict(aggregate["field_estimated_tokens"])
        aggregate["verdicts"] = dict(aggregate["verdicts"])
        aggregate["output_classes"] = dict(aggregate["output_classes"])
        actual = aggregate["actual_output_tokens"]
        projected = aggregate["projected_output_tokens"]
        reduction = max(0, actual - projected)
        aggregate["reduction_tokens"] = reduction
        aggregate["reduction_ratio"] = reduction / actual if actual else 0.0
        speed = actual / aggregate["actual_decode_time"] if aggregate["actual_decode_time"] else 0.0
        aggregate["projected_decode_time"] = projected / speed if speed else None
        aggregate["theoretical_decode_seconds_saved"] = (
            aggregate["actual_decode_time"] - aggregate["projected_decode_time"] if speed else None
        )

    facts = sum(
        len(story.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("events", []))
        for story in stories
    )
    return {
        "schema_version": "agent-pipeline-v3.1-generation-analysis-v1",
        "stories": len(stories),
        "facts": facts,
        "components": components,
        "call_rows": call_rows,
        "field_rows": field_rows,
        "projection_rows": projection_rows,
    }
