"""Deterministic accounting of span dispositions; no semantic inference."""

from __future__ import annotations


def account_coverage(target_ids, events, ignored_spans, non_event_span_ids) -> dict:
    """Compute coverage from validated dispositions without guessing missing semantics."""
    targets = list(target_ids)
    if len(targets) != len(set(targets)):
        raise ValueError("target片段ID重复")
    target_set = set(targets)
    disposition: dict[str, str] = {}

    def assign(span_id, kind):
        if span_id not in target_set:
            raise ValueError("处置引用了不存在的target片段")
        previous = disposition.get(span_id)
        if previous is not None and not (previous == kind == "event"):
            raise ValueError("同一片段出现重复处置")
        disposition[span_id] = kind

    for event in events:
        for span_id in event["source_ids"]:
            assign(span_id, "event")
    for ignored in ignored_spans:
        assign(ignored["source_id"], "ignored")
    for span_id in non_event_span_ids:
        assign(span_id, "non_event")

    groups = {kind: [span_id for span_id in targets if disposition.get(span_id) == kind]
              for kind in ("event", "ignored", "non_event")}
    uncovered = [span_id for span_id in targets if span_id not in disposition]
    return {
        "target_count": len(targets),
        "event_span_ids": groups["event"],
        "ignored_span_ids": groups["ignored"],
        "non_event_span_ids": groups["non_event"],
        "uncovered_span_ids": uncovered,
        "complete": not uncovered,
    }
