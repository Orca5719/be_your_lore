"""Extractor-only quality and performance accounting for Benchmark 4A."""

from __future__ import annotations

import json
import math


EVENT_FIELDS = ("actors", "event", "mental_state", "explicit", "modality", "conditions", "source_ids", "context_ids", "check_reason")


def _events(row: dict) -> list[dict]:
    return row.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("events", [])


def _key(event: dict) -> str:
    return json.dumps({field: event.get(field) for field in EVENT_FIELDS}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _score(cases: list[dict], rows: list[dict], labels: dict, mapping: dict) -> dict:
    by_id = {row["case_id"]: row for row in rows}
    gold_total = valid_total = event_total = hits = 0
    for case in cases:
        case_id = case["id"]
        events = _events(by_id[case_id]) if case_id in by_id else []
        event_ids = {event["id"] for event in events}
        case_labels = labels.get(case_id, {})
        case_mapping = mapping.get(case_id, {})
        gold_total += len(case.get("gold_facts", []))
        event_total += len(events)
        valid_total += sum(case_labels.get(event_id, {}).get("label") == "valid_checkable" for event_id in event_ids)
        for gold in case.get("gold_facts", []):
            if any(event_id in event_ids and case_labels.get(event_id, {}).get("label") == "valid_checkable"
                   for event_id in case_mapping.get(gold["id"], [])):
                hits += 1
    return {
        "recall": hits / gold_total if gold_total else None,
        "precision": valid_total / event_total if event_total else None,
        "gold_hits": hits, "gold_total": gold_total,
        "valid_events": valid_total, "emitted_events": event_total,
    }


def compare_extractions(cases: list[dict], source_rows: list[dict], candidate_rows: list[dict], review: dict) -> dict:
    source = {row["case_id"]: row for row in source_rows}
    old_labels = {key.removeprefix("candidate:"): value for key, value in review["system_event_labels"].items() if key.startswith("candidate:")}
    old_mapping = {key.removeprefix("candidate:"): value for key, value in review["event_mapping"].items() if key.startswith("candidate:")}
    baseline = _score(cases, source_rows, old_labels, old_mapping)
    labels: dict[str, dict] = {}
    mapping: dict[str, dict] = {}
    pending: list[dict] = []
    changed_cases = []
    for row in candidate_rows:
        case_id = row["case_id"]
        old_events = _events(source[case_id])
        old_by_key: dict[str, list[dict]] = {}
        for event in old_events:
            old_by_key.setdefault(_key(event), []).append(event)
        labels[case_id] = {}
        mapping[case_id] = {gold["id"]: [] for gold in next(case for case in cases if case["id"] == case_id).get("gold_facts", [])}
        new_events = _events(row)
        if sorted(_key(event) for event in old_events) != sorted(_key(event) for event in new_events):
            changed_cases.append(case_id)
        for event in new_events:
            matches = old_by_key.get(_key(event), [])
            if len(matches) != 1 or matches[0]["id"] not in old_labels.get(case_id, {}):
                pending.append({"case_id": case_id, "event_id": event["id"], "event": event})
                continue
            previous_id = matches[0]["id"]
            labels[case_id][event["id"]] = old_labels[case_id][previous_id]
            for gold_id, event_ids in old_mapping.get(case_id, {}).items():
                if previous_id in event_ids:
                    mapping[case_id].setdefault(gold_id, []).append(event["id"])
    candidate = _score(cases, candidate_rows, labels, mapping)
    if pending or len(candidate_rows) != len(cases):
        candidate["reviewed_lower_bound_recall"] = candidate["recall"]
        candidate["reviewed_lower_bound_precision"] = candidate["precision"]
        candidate["recall"] = None
        candidate["precision"] = None
    return {
        "baseline": baseline, "candidate": candidate,
        "pending_events": pending, "pending_count": len(pending),
        "changed_case_ids": changed_cases,
        "quality_status": "reviewed" if not pending and len(candidate_rows) == len(cases) else "pending_review",
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize_static(rows: list[dict], calls: list[dict], *, wall_seconds: float | None, batch_size: int) -> dict:
    total = lambda name: sum(float(call.get(name) or 0) for call in calls)
    output_tokens = int(total("output_tokens"))
    decode = total("decode_time")
    input_tokens = int(total("input_tokens"))
    prefill = total("prefill_time")
    attempts = [attempt for row in rows for window in row.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("calls", []) for attempt in window.get("attempts", [])]
    return {
        "batch_size": batch_size, "stories": len(rows),
        "events": sum(len(_events(row)) for row in rows),
        "status_counts": {status: sum(row.get("status") == status for row in rows) for status in ("ok", "partial", "error")},
        "llm_calls": len(calls),
        "retry_requests": sum(attempt.get("attempt", 1) > 1 and attempt.get("purpose") != "recover_missing_targets" for attempt in attempts),
        "coverage_recovery_requests": sum(attempt.get("purpose") == "recover_missing_targets" for attempt in attempts),
        "actual_avg_batch": sum(call["batch_size"] for call in calls) / len(calls) if calls else None,
        "input_tokens": input_tokens, "padded_input_tokens": int(total("padded_input_tokens")),
        "output_tokens": output_tokens,
        "prefill_seconds": prefill, "decode_seconds": decode, "llm_seconds": total("total_time"),
        "wall_seconds": wall_seconds,
        "ttft_p50_ms": _percentile([call["ttft_ms"] for call in calls if call.get("ttft_ms") is not None], 0.5),
        "ttft_p95_ms": _percentile([call["ttft_ms"] for call in calls if call.get("ttft_ms") is not None], 0.95),
        "prefill_useful_tokens_per_second": input_tokens / prefill if prefill else None,
        "decode_tokens_per_second": output_tokens / decode if decode else None,
        "stories_per_second": len(rows) / wall_seconds if wall_seconds else None,
        "peak_allocated": max((call.get("peak_allocated") or 0 for call in calls), default=None),
        "peak_reserved": max((call.get("peak_reserved") or 0 for call in calls), default=None),
    }


def render_table(summaries: list[dict], baseline: dict, *, failures: list[dict] | None = None,
                 baseline_performance: dict | None = None) -> str:
    def fmt(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"
    lines = [
        "# Benchmark 4A：Extractor 固定批处理", "",
        "仅重跑 Extractor；不是端到端成绩。提取结果与 Benchmark 3 原人工审核逐事件精确匹配；新增或变化事件待复核时不发布正式 Precision／Recall。", "",
        f"Benchmark 3 提取基线：Recall {fmt(baseline['recall'] * 100)}%，Precision {fmt(baseline['precision'] * 100)}%。", "",
        "| Batch | 实际平均 batch | Calls | 重试 / 覆盖恢复 | 输入 / Padding / 输出 tok | TTFT p50/p95 ms | Prefill / Decode / LLM / Wall s | Decode tok/s | stories/s | 峰值 allocated/reserved GiB | Recall / Precision | 待复核事件 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline_performance:
        lines[6:6] = [
            f"Benchmark 3 原顺序执行：{baseline_performance['calls']} 次调用，输出 {baseline_performance['output_tokens']} tokens，"
            f"Decode {fmt(baseline_performance['decode_seconds'])} s，LLM 总耗时 {fmt(baseline_performance['total_seconds'])} s。", "",
        ]
    for item in summaries:
        perf, quality = item["performance"], item["quality"]
        candidate = quality["candidate"]
        peak = lambda key: fmt(perf[key] / 2**30) if perf[key] is not None else "—"
        lines.append("| " + " | ".join([
            str(perf["batch_size"]), fmt(perf["actual_avg_batch"]), str(perf["llm_calls"]),
            f"{perf['retry_requests']} / {perf['coverage_recovery_requests']}",
            f"{perf['input_tokens']} / {perf['padded_input_tokens']} / {perf['output_tokens']}",
            f"{fmt(perf['ttft_p50_ms'])} / {fmt(perf['ttft_p95_ms'])}",
            f"{fmt(perf['prefill_seconds'])} / {fmt(perf['decode_seconds'])} / {fmt(perf['llm_seconds'])} / {fmt(perf['wall_seconds'])}",
            fmt(perf["decode_tokens_per_second"]), fmt(perf["stories_per_second"], 3),
            f"{peak('peak_allocated')} / {peak('peak_reserved')}",
            f"{fmt(candidate['recall'] * 100) if candidate['recall'] is not None else '待复核'} / {fmt(candidate['precision'] * 100) if candidate['precision'] is not None else '待复核'}",
            str(quality["pending_count"]),
        ]) + " |")
    for failure in failures or []:
        lines.append(f"| {failure['batch_size']} | OOM：{failure['error']} | " + " | ".join(["—"] * 10) + " |")
    lines += ["", "若某档位运行中断或恢复运行，Wall 时间只对应本次会话，不用于正式档位间对比。", ""]
    return "\n".join(lines)
