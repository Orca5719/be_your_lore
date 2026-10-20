"""Baseline Extractor adapter with a separate, programmatic coverage view."""

from __future__ import annotations

from agent_pipeline_v2.extractor import extract_events as baseline_extract_events

from .coverage import account_coverage


def extract_events(text: str, device: str = "auto", llm=None, progress=None) -> dict:
    result = baseline_extract_events(text, device=device, llm=llm, progress=progress)
    result["coverage_accounting"] = account_coverage(
        result["spans"], result["events"], result["ignored_spans"], result["non_event_span_ids"]
    )
    if set(result["coverage_accounting"]["uncovered_span_ids"]) != set(result["uncovered_span_ids"]):
        raise ValueError("程序覆盖账本与提取报告不一致")
    return result
