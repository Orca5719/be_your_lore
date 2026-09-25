"""Unified JSON and Markdown benchmark summary rendering."""

from __future__ import annotations

from typing import Any


def _get(mapping: Any, *keys: str, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_benchmark_summary(story_quality: dict, attribution: dict, judge_report: dict, run_status: dict) -> dict:
    detection = story_quality.get("detection", {})
    extraction = story_quality.get("extraction", {})
    retrieval = story_quality.get("retrieval", {})
    judge = story_quality.get("judge", {})
    results = [row for row in judge_report.get("results", []) if row.get("batch_size") == 8]
    performance = results[0] if results else {}
    return {
        "schema_version": "agent-pipeline-v2-benchmark-summary-v1",
        "headline": {
            "conflict_f1": _get(detection, "conflict", "f1"),
            "extraction_hallucination_rate": _get(extraction, "hallucination_rate", "value"),
            "extraction_recall": _get(extraction, "recall", "value"),
            "retrieval_recall_at_k": _get(retrieval, "recall_at_k", "value"),
            "judge_accuracy": _get(judge, "pipeline", "accuracy", "value"),
        },
        "quality": {
            "extraction": extraction,
            "retrieval": retrieval,
            "judge": judge,
            "detection": detection,
        },
        "errors": {"counts": attribution.get("counts", {}), "status": attribution.get("status", "unknown")},
        "performance": performance.get("distributions", {}) | {"batch_size": performance.get("batch_size", 8), "repeat_count": performance.get("repeat_count", 0)},
        "run_status": run_status,
    }


def render_benchmark_markdown(summary: dict) -> str:
    headline = summary.get("headline", {})
    performance = summary.get("performance", {})
    lines = [
        "# Agent Pipeline v2 Benchmark Summary",
        "",
        "## Headline Quality",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Conflict F1 | {_number(headline.get('conflict_f1'))} |",
        f"| Extraction Recall | {_number(headline.get('extraction_recall'))} |",
        f"| Extraction Hallucination Rate | {_number(headline.get('extraction_hallucination_rate'))} |",
        f"| Retrieval Recall@K | {_number(headline.get('retrieval_recall_at_k'))} |",
        f"| Judge Accuracy | {_number(headline.get('judge_accuracy'))} |",
        "",
        "## Error Taxonomy",
        "",
        "| Error Type | Count |",
        "|---|---:|",
    ]
    for name, count in sorted(summary.get("errors", {}).get("counts", {}).items()):
        lines.append(f"| {name} | {_number(count, 0)} |")
    if not summary.get("errors", {}).get("counts"):
        lines.append("| N/A | N/A |")
    lines.extend([
        "",
        "## Batch-8 Performance",
        "",
        "| Metric | Median | Min | Max | Stddev |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("total_judge_seconds", "facts_per_second", "peak_allocated_gib", "peak_reserved_gib"):
        row = performance.get(name, {})
        lines.append(f"| {name} | {_number(row.get('median'))} | {_number(row.get('min'))} | {_number(row.get('max'))} | {_number(row.get('stddev'))} |")
    lines.extend(["", f"Batch size: {_number(performance.get('batch_size'), 0)}; repeats: {_number(performance.get('repeat_count'), 0)}.", ""])
    return "\n".join(lines)
