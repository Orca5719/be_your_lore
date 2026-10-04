import pytest

from agent_pipeline_v4.continuous_report import (summarize_continuous, static_wasted_steps,
                                                  compare_case_outputs, render_continuous_table)


def test_step_times_are_counted_once_even_when_shared_by_requests():
    steps = [
        {"kind": "prefill", "batch_size": 1, "seconds": 0.1, "input_tokens": 10,
         "padded_input_tokens": 0, "output_tokens": 1, "peak_allocated": 100,
         "peak_reserved": 120},
        {"kind": "prefill", "batch_size": 1, "seconds": 0.2, "input_tokens": 12,
         "padded_input_tokens": 0, "output_tokens": 1, "peak_allocated": 110,
         "peak_reserved": 130},
        {"kind": "decode", "batch_size": 2, "seconds": 0.3, "input_tokens": 0,
         "padded_input_tokens": 0, "output_tokens": 2, "peak_allocated": 115,
         "peak_reserved": 140},
    ]
    requests = [{"ttft_ms": 100, "generated_tokens": 2, "queue_wait_ms": 0},
                {"ttft_ms": 200, "generated_tokens": 2, "queue_wait_ms": 5}]
    rows = [{"status": "ok", "result": {"benchmark_stages": {"extraction": {
        "events": [], "calls": []}}}}] * 2
    summary = summarize_continuous(rows, steps, requests, capacity=2, wall_seconds=0.8,
                                   scheduler={"active_slot_steps": 2, "capacity_slot_steps": 2,
                                              "finished_request_waste_steps": 0})
    assert summary["llm_seconds"] == pytest.approx(0.6)
    assert summary["non_model_overhead_seconds"] == pytest.approx(0.2)
    assert summary["decode_seconds"] == 0.3
    assert summary["output_tokens"] == 4
    assert summary["slot_utilization"] == 1.0
    assert summary["ttft_p95_ms"] == 200
    assert summary["peak_allocated"] == 115


def test_static_finished_rows_count_as_wasted_decode_steps():
    calls = [{"row_output_tokens": [2, 5, 3]},
             {"row_output_tokens": [4]}]
    assert static_wasted_steps(calls) == 5


def test_case_diff_reports_changed_decoded_output_and_event_status():
    def row(case_id, raw, event, status="ok"):
        return {"case_id": case_id, "status": status, "result": {"benchmark_stages": {"extraction": {
            "events": [{"event": event}], "calls": [{"window_id": 1, "attempts": [
                {"raw_output": raw, "timing": {"generated_tokens": len(raw)}}]}]}}}}

    differences = compare_case_outputs([row("A", "old", "first"), row("B", "same", "same")],
                                       [row("A", "new", "second", "partial"), row("B", "same", "same")])
    assert len(differences) == 1
    assert differences[0]["case_id"] == "A"
    assert differences[0]["decoded_output_changed"] is True
    assert differences[0]["event_changed"] is True
    assert differences[0]["status_changed"] is True


def test_markdown_distinguishes_request_ttft_and_static_batch_ttft():
    rows = [{"status": "partial", "result": {"benchmark_stages": {"extraction": {
        "events": [], "calls": []}}}}]
    steps = [{"kind": "prefill", "seconds": .2, "output_tokens": 1, "batch_size": 1,
              "input_tokens": 10, "padded_input_tokens": 0,
              "peak_allocated": None, "peak_reserved": None}]
    requests = [{"ttft_ms": 200, "queue_wait_ms": 10}]
    performance = summarize_continuous(rows, steps, requests, capacity=1,
                                       wall_seconds=.3, scheduler=None)
    report = render_continuous_table([{"performance": performance,
                                       "quality": {"candidate": {"recall": None, "precision": None},
                                                   "pending_count": 1},
                                       "difference_case_ids": ["SL-010"]}],
                                     {1: {"performance": {"wall_seconds": 2.0},
                                          "wasted_finished_steps": 0}})
    assert "请求级 TTFT" in report
    assert "SL-010" in report
    assert "待复核" in report
