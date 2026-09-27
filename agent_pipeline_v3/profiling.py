from __future__ import annotations

import math
import time
from typing import Callable


COMPONENTS = frozenset({"extractor", "judge"})
STATUSES = frozenset({"ok", "error"})
SCHEMA_VERSION = "agent-pipeline-v3-inference-call-v1"


def _nonnegative_number(value, field: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")


def validate_call_record(record: dict) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid profiling schema_version")
    if record.get("component") not in COMPONENTS:
        raise ValueError("invalid component")
    if record.get("status") not in STATUSES:
        raise ValueError("invalid status")
    if not isinstance(record.get("call_id"), str) or not record["call_id"].startswith("LLM-"):
        raise ValueError("invalid call_id")
    if record.get("story_id") is not None and not isinstance(record["story_id"], str):
        raise ValueError("invalid story_id")
    for field in ("batch_size", "input_tokens", "padded_input_tokens", "compute_input_tokens", "output_tokens"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if field == "batch_size" else 0):
            raise ValueError(f"invalid {field}")
    if record["compute_input_tokens"] != record["input_tokens"] + record["padded_input_tokens"]:
        raise ValueError("compute_input_tokens does not match useful plus padding")
    for field in ("prefill_time", "decode_time", "ttft_ms", "peak_allocated", "peak_reserved"):
        _nonnegative_number(record.get(field), field, nullable=True)
    _nonnegative_number(record.get("total_time"), "total_time")
    if record["prefill_time"] is None or record["decode_time"] is None:
        if record["prefill_time"] is not None or record["decode_time"] is not None or record["ttft_ms"] is not None:
            raise ValueError("phase times must be all present or all null")
    else:
        if abs((record["prefill_time"] + record["decode_time"]) - record["total_time"]) > 1e-6:
            raise ValueError("phase times do not sum to total_time")
        if abs(record["ttft_ms"] - record["prefill_time"] * 1000) > 1e-3:
            raise ValueError("ttft_ms does not match prefill_time")
    if record["status"] == "error" and not isinstance(record.get("error"), str):
        raise ValueError("error call must include error text")


class TraceCollector:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.story_id: str | None = None
        self.records: list[dict] = []
        self._sequence = 0

    def set_story(self, story_id: str | None) -> None:
        if story_id is not None and (not isinstance(story_id, str) or not story_id.strip()):
            raise ValueError("story_id must be nonempty text or null")
        self.story_id = story_id

    def clear(self) -> None:
        self.records.clear()

    def start(self, component: str, batch_size: int, input_tokens: int, padded_input_tokens: int):
        if component not in COMPONENTS:
            raise ValueError("invalid component")
        for value, name, minimum in (
            (batch_size, "batch_size", 1),
            (input_tokens, "input_tokens", 0),
            (padded_input_tokens, "padded_input_tokens", 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid {name}")
        self._sequence += 1
        return CallSpan(
            collector=self,
            call_id=f"LLM-{self._sequence:06d}",
            story_id=self.story_id,
            component=component,
            batch_size=batch_size,
            input_tokens=input_tokens,
            padded_input_tokens=padded_input_tokens,
            started=self.clock(),
        )


class CallSpan:
    def __init__(self, *, collector, call_id, story_id, component, batch_size, input_tokens, padded_input_tokens, started):
        self.collector = collector
        self.call_id = call_id
        self.story_id = story_id
        self.component = component
        self.batch_size = batch_size
        self.input_tokens = input_tokens
        self.padded_input_tokens = padded_input_tokens
        self.started = started
        self.first_token_ready: float | None = None
        self.finished = False

    def mark_first_token(self) -> None:
        if self.first_token_ready is None:
            self.first_token_ready = self.collector.clock()

    def finish(
        self,
        output_tokens: int,
        *,
        status: str = "ok",
        peak_allocated: int | None = None,
        peak_reserved: int | None = None,
        error: str | None = None,
    ) -> dict:
        if self.finished:
            raise RuntimeError("profiling span already finished")
        self.finished = True
        ended = self.collector.clock()
        total = max(0.0, ended - self.started)
        prefill = None if self.first_token_ready is None else max(0.0, self.first_token_ready - self.started)
        decode = None if prefill is None else max(0.0, total - prefill)
        record = {
            "schema_version": SCHEMA_VERSION,
            "call_id": self.call_id,
            "story_id": self.story_id,
            "component": self.component,
            "status": status,
            "input_tokens": self.input_tokens,
            "padded_input_tokens": self.padded_input_tokens,
            "compute_input_tokens": self.input_tokens + self.padded_input_tokens,
            "output_tokens": output_tokens,
            "prefill_time": prefill,
            "decode_time": decode,
            "total_time": total,
            "ttft_ms": None if prefill is None else prefill * 1000,
            "batch_size": self.batch_size,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "error": error,
        }
        validate_call_record(record)
        self.collector.records.append(record)
        return record


class FirstTokenTimer:
    """Stopping criterion that timestamps the first completed generation step."""

    def __init__(self, span: CallSpan, synchronize: Callable[[], None] | None = None):
        self.span = span
        self.synchronize = synchronize
        self.marked = False

    def __call__(self, input_ids, scores, **kwargs):
        if not self.marked:
            if self.synchronize is not None:
                self.synchronize()
            self.span.mark_first_token()
            self.marked = True
        try:
            import torch

            dtype = torch.bool
        except ModuleNotFoundError:
            dtype = None
        return input_ids.new_zeros((input_ids.shape[0],), dtype=dtype)
