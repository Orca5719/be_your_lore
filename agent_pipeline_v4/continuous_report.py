"""Benchmark 4B accounting; model steps and request latency stay separate."""

from __future__ import annotations

import math
import json


def _percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def static_wasted_steps(calls: list[dict]) -> int:
    """Static generate keeps finished rows padded until the longest row ends."""
    return sum(sum(max(lengths) - length for length in lengths)
               for call in calls if (lengths := call.get("row_output_tokens", [])))


def compare_case_outputs(static_rows: list[dict], continuous_rows: list[dict]) -> list[dict]:
    """Compare what the saved 4A/4B artifacts actually contain, by story."""
    old = {row["case_id"]: row for row in static_rows}
    result = []
    for row in continuous_rows:
        case_id = row["case_id"]
        previous = old.get(case_id)
        if previous is None:
            result.append({"case_id": case_id, "status_changed": True,
                           "event_changed": True, "decoded_output_changed": True,
                           "note": "missing in static source"})
            continue

        def extraction(value):
            return value["result"]["benchmark_stages"]["extraction"]

        def outputs(value):
            return [[{"text": attempt.get("raw_output", ""),
                      "generated_tokens": attempt.get("timing", {}).get("generated_tokens")}
                     for attempt in window.get("attempts", [])]
                    for window in extraction(value).get("calls", [])]

        prior_outputs, current_outputs = outputs(previous), outputs(row)
        old_events = json.dumps(extraction(previous).get("events", []), ensure_ascii=False, sort_keys=True)
        new_events = json.dumps(extraction(row).get("events", []), ensure_ascii=False, sort_keys=True)
        status_changed = previous.get("status") != row.get("status")
        event_changed = old_events != new_events
        output_changed = prior_outputs != current_outputs
        if status_changed or event_changed or output_changed:
            result.append({"case_id": case_id, "status_changed": status_changed,
                           "event_changed": event_changed,
                           "decoded_output_changed": output_changed,
                           "static_status": previous.get("status"),
                           "continuous_status": row.get("status"),
                           "static_event_count": len(extraction(previous).get("events", [])),
                           "continuous_event_count": len(extraction(row).get("events", [])),
                           "static_outputs": prior_outputs,
                           "continuous_outputs": current_outputs})
    return result


def summarize_continuous(rows: list[dict], steps: list[dict], requests: list[dict],
                         *, capacity: int, wall_seconds: float | None, scheduler: dict | None) -> dict:
    prefill = sum(float(step["seconds"]) for step in steps if step["kind"] == "prefill")
    decode = sum(float(step["seconds"]) for step in steps if step["kind"] == "decode")
    llm = prefill + decode
    output_tokens = sum(int(step["output_tokens"]) for step in steps)
    input_tokens = sum(int(step.get("input_tokens", 0)) for step in steps)
    decode_steps = [step for step in steps if step["kind"] == "decode"]
    active_steps = (scheduler or {}).get("active_slot_steps")
    if active_steps is None:
        active_steps = sum(step["batch_size"] for step in decode_steps)
    capacity_steps = (scheduler or {}).get("capacity_slot_steps")
    if capacity_steps is None:
        capacity_steps = capacity * len(decode_steps)
    attempts = [attempt for row in rows for window in row.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("calls", [])
                for attempt in window.get("attempts", [])]
    events = sum(len(row.get("result", {}).get("benchmark_stages", {}).get("extraction", {}).get("events", [])) for row in rows)
    return {
        "capacity": capacity, "stories": len(rows), "events": events,
        "status_counts": {status: sum(row.get("status") == status for row in rows)
                          for status in ("ok", "partial", "error")},
        "requests": len(requests), "model_steps": len(steps),
        "retry_requests": sum(attempt.get("attempt", 1) > 1 and attempt.get("purpose") != "recover_missing_targets"
                              for attempt in attempts),
        "coverage_recovery_requests": sum(attempt.get("purpose") == "recover_missing_targets" for attempt in attempts),
        "actual_avg_decode_batch": active_steps / len(decode_steps) if decode_steps else None,
        "input_tokens": input_tokens,
        "padded_input_tokens": sum(int(step.get("padded_input_tokens", 0)) for step in steps),
        "output_tokens": output_tokens,
        "prefill_seconds": prefill, "decode_seconds": decode, "llm_seconds": llm,
        "wall_seconds": wall_seconds,
        "non_model_overhead_seconds": max(0.0, wall_seconds - llm) if wall_seconds is not None else None,
        "scheduler_seconds": sum(float(step.get("scheduler_seconds", 0)) for step in steps),
        "ttft_p50_ms": _percentile([float(row["ttft_ms"]) for row in requests if row.get("ttft_ms") is not None], .5),
        "ttft_p95_ms": _percentile([float(row["ttft_ms"]) for row in requests if row.get("ttft_ms") is not None], .95),
        "queue_wait_p50_ms": _percentile([float(row["queue_wait_ms"]) for row in requests if row.get("queue_wait_ms") is not None], .5),
        "queue_wait_p95_ms": _percentile([float(row["queue_wait_ms"]) for row in requests if row.get("queue_wait_ms") is not None], .95),
        "decode_tokens_per_second": sum(step["output_tokens"] for step in decode_steps) / decode if decode else None,
        "stories_per_second": len(rows) / wall_seconds if wall_seconds else None,
        "slot_utilization": active_steps / capacity_steps if capacity_steps else None,
        "active_slot_steps": active_steps, "capacity_slot_steps": capacity_steps,
        "finished_request_waste_steps": 0,
        "peak_allocated": max((step.get("peak_allocated") or 0 for step in steps), default=None),
        "peak_reserved": max((step.get("peak_reserved") or 0 for step in steps), default=None),
    }


def render_continuous_table(items: list[dict], static: dict[int, dict], *, failures: list[dict] | None = None) -> str:
    def fmt(value, places=2):
        return "—" if value is None else f"{value:.{places}f}"

    lines = ["# Benchmark 4B：Extractor 连续批处理", "",
             "仅重跑 Benchmark 3 Extractor。请求级 TTFT 从单请求 prefill 前开始，不含排队；排队时间单列。"
             "模型时间按 forward 步骤只计一次，不累加多个请求共享的 decode 时间。", "",
             "| 容量 | 处理状态 | 请求 / forward 步 | 平均decode batch | 输入 / padding / 输出 tok | 请求TTFT p50/p95 ms | Prefill / Decode / Wall s | Decode tok/s | 槽位利用率 | 已结束请求无效步 | 峰值allocated/reserved GiB | Recall / Precision | 待复核 | 对应4A Wall / 无效步 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in items:
        p, q = item["performance"], item["quality"]
        candidate = q["candidate"]
        baseline = static.get(p["capacity"])
        b_perf = baseline["performance"] if baseline else None
        wasted = baseline.get("wasted_finished_steps") if baseline else None
        peak = lambda name: fmt(p[name] / 2**30) if p[name] is not None else "—"
        lines.append("| " + " | ".join([
            str(p["capacity"]), f"{p['status_counts']['ok']}/{p['stories']} ok",
            f"{p['requests']} / {p['model_steps']}", fmt(p["actual_avg_decode_batch"]),
            f"{p['input_tokens']} / {p['padded_input_tokens']} / {p['output_tokens']}",
            f"{fmt(p['ttft_p50_ms'])} / {fmt(p['ttft_p95_ms'])}",
            f"{fmt(p['prefill_seconds'])} / {fmt(p['decode_seconds'])} / {fmt(p['wall_seconds'])}",
            fmt(p["decode_tokens_per_second"]),
            fmt(p["slot_utilization"] * 100) + "%" if p["slot_utilization"] is not None else "—",
            str(p["finished_request_waste_steps"]),
            f"{peak('peak_allocated')} / {peak('peak_reserved')}",
            f"{fmt(candidate['recall'] * 100) if candidate['recall'] is not None else '待复核'} / "
            f"{fmt(candidate['precision'] * 100) if candidate['precision'] is not None else '待复核'}",
            str(q["pending_count"]),
            f"{fmt(b_perf['wall_seconds']) if b_perf else '—'} / {wasted if wasted is not None else '—'}",
        ]) + " |")
    lines += ["", "`non_model_overhead_seconds` 是 Wall 减模型 forward 时间，含 tokenization、KV cache 整理、调度、校验和保存；不冒充纯调度耗时。"
              "请求级 TTFT 与 4A 批次级 TTFT 不直接比较。", "",
              "新/变更事件需复核；partial 或 error 均保留原样，不自动重试整篇来美化质量。", ""]
    lines += ["| 容量 | 格式重试 / 缺口恢复 | 排队 p50/p95 ms | 非模型开销 / KV整理 s | stories/s | 与4A输出不同的故事 |",
              "|---:|---:|---:|---:|---:|---|"]
    for item in items:
        p = item["performance"]
        changed = item.get("difference_case_ids", [])
        lines.append("| " + " | ".join([
            str(p["capacity"]), f"{p['retry_requests']} / {p['coverage_recovery_requests']}",
            f"{fmt(p['queue_wait_p50_ms'])} / {fmt(p['queue_wait_p95_ms'])}",
            f"{fmt(p['non_model_overhead_seconds'])} / {fmt(p['scheduler_seconds'])}",
            fmt(p["stories_per_second"], 3),
            ", ".join(changed) if changed else "无",
        ]) + " |")
    lines += ["", "4A未保存原始生成token ID，逐案对照只比较解码文本与已记录的生成token数；"
              "`case_differences.json`保留每个变化故事的窗口输出和事件差异。", ""]
    for failure in failures or []:
        lines.append(f"容量 {failure['capacity']}：{failure['status']} — {failure['error']}")
    if failures:
        lines.append("")
    return "\n".join(lines)
