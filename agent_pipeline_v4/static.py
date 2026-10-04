"""Batch the frozen Benchmark 3 extractor's synchronous generation calls."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import time

from qwen_judge import GenerationError
from agent_pipeline_v2.extractor import extract_events
from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
from agent_pipeline_v2_2.runner import write_jsonl_atomic


@dataclass
class _Request:
    story_id: str
    messages: list[dict]
    max_new_tokens: int
    future: Future


class StaticBatcher:
    def __init__(self, model, batch_size: int, *, fill_wait_seconds: float = 0.02):
        if batch_size not in (1, 2, 4, 8):
            raise ValueError("extractor batch size must be 1, 2, 4, or 8")
        self.model = model
        self.batch_size = batch_size
        self.fill_wait_seconds = fill_wait_seconds
        self.queue: Queue[_Request | None] = Queue()
        self.records: list[dict] = []
        self._thread: Thread | None = None
        self._closed = False
        self._lock = Lock()
        self.fatal_error: Exception | None = None

    def __enter__(self):
        self._thread = Thread(target=self._serve, name="extractor-static-batcher", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        with self._lock:
            self._closed = True
        self.queue.put(None)
        self._thread.join()

    def submit(self, story_id: str, messages: list[dict], max_new_tokens: int):
        with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")
            future = Future()
            self.queue.put(_Request(story_id, messages, max_new_tokens, future))
        return future.result()

    def _serve(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.fill_wait_seconds
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get(timeout=max(0.0, deadline - time.perf_counter()))
                except Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                if item.max_new_tokens != first.max_new_tokens:
                    self.queue.put(item)
                    break
                batch.append(item)
            if self.fatal_error is not None:
                for item in batch:
                    item.future.set_result((None, {}, self.fatal_error))
                continue
            self._run_batch(batch)

    def _run_batch(self, batch: list[_Request]):
        try:
            self.model.last_batch_generation = {}
            outputs = self.model._generate_batch(
                [item.messages for item in batch], max_new_tokens=batch[0].max_new_tokens
            )
            if len(outputs) != len(batch):
                raise RuntimeError("batch model returned the wrong number of rows")
            timing = dict(self.model.last_batch_generation)
            record = timing.get("profile")
            if not isinstance(record, dict):
                raise RuntimeError("batch model omitted its profiling record")
            if len(timing.get("generated_tokens", [])) != len(batch) or len(timing.get("input_tokens", [])) != len(batch):
                raise RuntimeError("batch model omitted row-level token counts")
            record["component"] = "extractor"
            record["story_id"] = None
            record["story_ids"] = [item.story_id for item in batch]
            record["row_output_tokens"] = list(timing["generated_tokens"])
            record["row_input_tokens"] = list(timing["input_tokens"])
            self.records.append(record)
            truncated = set(timing.get("truncated_indices", []))
            for index, (item, raw) in enumerate(zip(batch, outputs)):
                row_timing = {
                    "call_id": timing["call_id"],
                    "batch_size": len(batch),
                    "input_tokens": timing["input_tokens"][index],
                    "generated_tokens": timing["generated_tokens"][index],
                    "seconds": timing["seconds"],
                    "ttft_ms": record.get("ttft_ms"),
                    "decode_seconds": record.get("decode_time"),
                    "profile": record,
                }
                error = GenerationError("生成达到上限且未结束", raw) if index in truncated else None
                item.future.set_result((raw, row_timing, error))
        except Exception as exc:
            timing = getattr(self.model, "last_batch_generation", {})
            record = timing.get("profile") if isinstance(timing, dict) else None
            if isinstance(record, dict):
                record["component"] = "extractor"
                record["story_id"] = None
                record["story_ids"] = [item.story_id for item in batch]
                self.records.append(record)
            if "out of memory" in str(exc).lower():
                self.fatal_error = exc
            for item in batch:
                if not item.future.done():
                    item.future.set_result((None, {"call_id": record.get("call_id")} if isinstance(record, dict) else {}, exc))


class QueuedLLM:
    """One per story, so the frozen repair wrapper keeps row-local timings."""

    def __init__(self, batcher: StaticBatcher, story_id: str):
        self.batcher = batcher
        self.story_id = story_id
        self.device = batcher.model.device
        self.load_seconds = batcher.model.load_seconds
        self.last_generation: dict = {}

    def _generate(self, messages, max_new_tokens=1536):
        raw, timing, error = self.batcher.submit(self.story_id, messages, max_new_tokens)
        self.last_generation = timing
        if error is not None:
            raise error
        return raw


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_single_story(model, text: str) -> bool:
    """Detect a batch-one semantic difference before starting formal measurements."""
    if hasattr(model, "set_story_id"):
        model.set_story_id("SMOKE")
    original = extract_events(text, device=model.device, llm=ExtractionRepairLLM(model))
    with StaticBatcher(model, 1) as batcher:
        batched = extract_events(text, device=model.device,
                                 llm=ExtractionRepairLLM(QueuedLLM(batcher, "SMOKE")))
    fields = ("status", "events", "ignored_spans", "non_event_span_ids", "uncovered_span_ids")
    if any(original[field] != batched[field] for field in fields):
        raise RuntimeError("Benchmark 3 single-row and Benchmark 4 batch-one extraction differ; formal run stopped")
    if hasattr(model, "clear_traces"):
        model.clear_traces()
    return True


def run_static_cases(cases: list[dict], model, batch_size: int, output: Path, *, warmup: bool = True, progress=None) -> dict:
    """Run the unchanged extraction routine concurrently; only generation is batched."""
    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "extractor_runs.jsonl"
    call_path = output / "inference_calls.jsonl"
    rows = _read_jsonl(row_path)
    calls = _read_jsonl(call_path)
    expected = [case["id"] for case in cases]
    saved_ids = [row.get("case_id") for row in rows]
    if len(saved_ids) != len(set(saved_ids)) or any(case_id not in expected for case_id in saved_ids):
        raise ValueError("saved extractor rows contain duplicate or unknown stories")
    rows = [row for row in rows if row.get("status") == "ok"]
    if hasattr(model, "collector"):
        model.collector.resume_after(calls)
    remaining = [case for case in cases if case["id"] not in {row["case_id"] for row in rows}]
    if not remaining:
        return {"rows": sorted(rows, key=lambda row: expected.index(row["case_id"])), "calls": calls, "wall_seconds": None}

    if warmup:
        with StaticBatcher(model, batch_size) as batcher:
            proxy = ExtractionRepairLLM(QueuedLLM(batcher, "WARMUP"))
            extract_events(cases[0]["story"], device=model.device, llm=proxy)
        if batcher.fatal_error is not None:
            raise RuntimeError(f"batch size {batch_size} warmup failed: {batcher.fatal_error}") from batcher.fatal_error
        if hasattr(model, "clear_traces"):
            model.clear_traces()

    def run_one(case: dict, batcher: StaticBatcher) -> dict:
        started = time.perf_counter()
        proxy = ExtractionRepairLLM(QueuedLLM(batcher, case["id"]))
        try:
            extraction = extract_events(case["story"], device=model.device, llm=proxy)
            status = extraction["status"]
            error = None
        except Exception as exc:
            extraction = {"status": "error", "events": [], "calls": [], "uncovered_span_ids": []}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "case_id": case["id"], "system": "benchmark-4-static-extractor", "status": status,
            "result": {"judge": {"events": extraction["events"], "items": []},
                       "benchmark_stages": {"extraction": extraction}},
            "stage_metrics": {"extraction": {"seconds": time.perf_counter() - started}},
        }
        if error:
            row["error"] = error
        return row

    started = time.perf_counter()
    persisted_batch_records = 0
    with StaticBatcher(model, batch_size) as batcher:
        with ThreadPoolExecutor(max_workers=min(len(remaining), max(8, batch_size * 2))) as pool:
            futures = {pool.submit(run_one, case, batcher): case["id"] for case in remaining}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda value: expected.index(value["case_id"]))
                calls.extend(batcher.records[persisted_batch_records:])
                persisted_batch_records = len(batcher.records)
                write_jsonl_atomic(row_path, rows)
                write_jsonl_atomic(call_path, calls)
                if progress:
                    progress(len(rows), len(cases), row["case_id"])
    if batcher.fatal_error is not None:
        raise RuntimeError(f"batch size {batch_size} failed: {batcher.fatal_error}") from batcher.fatal_error
    return {"rows": rows, "calls": calls, "wall_seconds": time.perf_counter() - started}
