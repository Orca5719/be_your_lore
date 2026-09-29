import pytest

from agent_pipeline_v3.profiling import FirstTokenTimer, TraceCollector, validate_call_record


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FalseRows:
    def __init__(self, count):
        self.count = count

    def tolist(self):
        return [False] * self.count


class FakeInputIds:
    def __init__(self, rows, width):
        self.shape = (rows, width)

    def new_zeros(self, shape, dtype=None):
        return FalseRows(shape[0])


def test_trace_collector_records_complete_call_and_identity():
    collector = TraceCollector(clock=Clock(10.0, 10.8, 12.5))
    collector.set_story("SL-001")
    span = collector.start("extractor", batch_size=1, input_tokens=100, padded_input_tokens=0)
    span.mark_first_token()
    record = span.finish(output_tokens=20, peak_allocated=123, peak_reserved=456)

    assert record["call_id"] == "LLM-000001"
    assert record["story_id"] == "SL-001"
    assert record["component"] == "extractor"
    assert record["prefill_time"] == pytest.approx(0.8)
    assert record["decode_time"] == pytest.approx(1.7)
    assert record["total_time"] == pytest.approx(2.5)
    assert record["ttft_ms"] == pytest.approx(800.0)
    assert record["compute_input_tokens"] == 100
    validate_call_record(record)


def test_error_before_first_token_has_explicit_null_phase_times():
    collector = TraceCollector(clock=Clock(3.0, 3.4))
    span = collector.start("judge", batch_size=2, input_tokens=17, padded_input_tokens=3)
    record = span.finish(output_tokens=0, status="error", error="boom")

    assert record["prefill_time"] is None
    assert record["decode_time"] is None
    assert record["total_time"] == pytest.approx(0.4)
    assert record["ttft_ms"] is None
    assert record["compute_input_tokens"] == 20
    validate_call_record(record)


def test_trace_ids_are_monotonic_and_clear_keeps_sequence_unique():
    collector = TraceCollector(clock=Clock(0, 1, 2, 3))
    first = collector.start("extractor", 1, 1, 0).finish(0, status="error", error="x")
    collector.clear()
    second = collector.start("judge", 1, 1, 0).finish(0, status="error", error="x")
    assert first["call_id"] == "LLM-000001"
    assert second["call_id"] == "LLM-000002"
    assert collector.records == [second]


def test_resume_after_existing_records_prevents_duplicate_call_ids():
    collector = TraceCollector(clock=Clock(0, 1))
    collector.resume_after([{"call_id": "LLM-000041"}, {"call_id": "LLM-000009"}])
    record = collector.start("extractor", 1, 1, 0).finish(0, status="error", error="x")
    assert record["call_id"] == "LLM-000042"


def test_first_token_timer_marks_only_once_and_never_stops_generation():
    collector = TraceCollector(clock=Clock(0.0, 0.25, 1.0))
    span = collector.start("judge", 2, 10, 2)
    sync_calls = []
    timer = FirstTokenTimer(span, synchronize=lambda: sync_calls.append("sync"))

    first = timer(FakeInputIds(2, 4), None)
    second = timer(FakeInputIds(2, 5), None)
    record = span.finish(4)

    assert first.tolist() == [False, False]
    assert second.tolist() == [False, False]
    assert second is first
    assert sync_calls == ["sync"]
    assert record["prefill_time"] == pytest.approx(0.25)


def test_validation_rejects_inconsistent_phase_sum():
    collector = TraceCollector(clock=Clock(0.0, 1.0, 2.0))
    span = collector.start("extractor", 1, 1, 0)
    span.mark_first_token()
    record = span.finish(1)
    record["decode_time"] = 9.0
    with pytest.raises(ValueError, match="phase times"):
        validate_call_record(record)


def test_validation_rejects_unknown_component():
    collector = TraceCollector(clock=Clock(0.0, 1.0))
    record = collector.start("extractor", 1, 1, 0).finish(0, status="error", error="x")
    record["component"] = "report"
    with pytest.raises(ValueError, match="component"):
        validate_call_record(record)
