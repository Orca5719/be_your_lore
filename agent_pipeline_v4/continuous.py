"""Request-level continuous decode for the frozen Benchmark 3 extractor."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock, Thread
import time
import json
from pathlib import Path

from agent_pipeline_v2.extractor import extract_events
from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
from agent_pipeline_v2_2.runner import write_jsonl_atomic

from qwen_judge import GenerationError


@dataclass
class _Request:
    story_id: str
    messages: list[dict]
    max_new_tokens: int
    future: Future
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass
class _Active:
    request: _Request
    state: object
    tokens: list[int]
    input_tokens: int
    first_token_seconds: float
    admitted_at: float
    decode_seconds: float = 0.0
    peak_allocated: int | None = None
    peak_reserved: int | None = None


class ContinuousEngine:
    """Model boundary: prefill one arrival; decode current live rows together."""

    eos_ids: set[int]

    def prefill(self, request: _Request) -> tuple[object, int, dict]:
        raise NotImplementedError

    def decode(self, states: list[object], last_tokens: list[int]) -> tuple[list[int], dict]:
        raise NotImplementedError

    def text(self, tokens: list[int]) -> str:
        raise NotImplementedError


class ContinuousBatcher:
    def __init__(self, engine: ContinuousEngine, capacity: int, *, fill_wait_seconds: float = 0.02,
                 step_offset: int = 0, request_offset: int = 0):
        if capacity not in (1, 2, 4, 8):
            raise ValueError("continuous capacity must be 1, 2, 4, or 8")
        self.engine = engine
        self.capacity = capacity
        self.fill_wait_seconds = fill_wait_seconds
        self.queue: Queue[_Request | None] = Queue()
        self.records: list[dict] = []
        self.steps: list[dict] = []
        self.metrics = {"active_slot_steps": 0, "capacity_slot_steps": 0,
                        "finished_request_waste_steps": 0, "scheduler_seconds": 0.0}
        self.fatal_error: Exception | None = None
        self._closed = False
        self._thread: Thread | None = None
        self._lock = Lock()
        self._next_id = request_offset
        self._step_offset = step_offset

    def __enter__(self):
        self._thread = Thread(target=self._serve, name="extractor-continuous-batcher", daemon=True)
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
                raise RuntimeError("continuous batcher is closed")
            if self.fatal_error is not None:
                raise self.fatal_error
            future = Future()
            self.queue.put(_Request(story_id, messages, max_new_tokens, future))
        return future.result()

    def _id(self):
        self._next_id += 1
        return f"CB-{self._next_id:06d}"

    def _step(self, kind: str, story_ids: list[str], metrics: dict):
        self.steps.append({"step_id": f"STEP-{self._step_offset+len(self.steps)+1:06d}", "kind": kind,
                           "story_ids": list(story_ids), "batch_size": len(story_ids),
                           "seconds": float(metrics["seconds"]),
                           "input_tokens": int(metrics.get("input_tokens", 0)),
                           "padded_input_tokens": int(metrics.get("padded_input_tokens", 0)),
                           "output_tokens": len(story_ids),
                           "scheduler_seconds": float(metrics.get("scheduler_seconds", 0)),
                           "peak_allocated": metrics.get("peak_allocated"),
                           "peak_reserved": metrics.get("peak_reserved")})

    def _finish(self, row: _Active):
        try:
            raw = self.engine.text(row.tokens)
        except Exception as exc:
            row.request.future.set_result((None, {}, exc))
            return
        generated = len(row.tokens)
        error = None
        if generated >= row.request.max_new_tokens and row.tokens[-1] not in self.engine.eos_ids:
            error = GenerationError("生成达到上限且未结束", raw)
        elapsed = row.first_token_seconds + row.decode_seconds
        timing = {"call_id": self._id(), "batch_size": 1, "input_tokens": row.input_tokens,
                  "generated_tokens": generated, "seconds": elapsed,
                  "ttft_ms": row.first_token_seconds * 1000,
                  "decode_seconds": row.decode_seconds,
                  "queue_wait_ms": (row.admitted_at - row.request.enqueued_at) * 1000}
        self.records.append({"call_id": timing["call_id"], "story_id": row.request.story_id,
                             "status": "truncated" if error else "ok", **timing,
                             "peak_allocated": row.peak_allocated,
                             "peak_reserved": row.peak_reserved})
        row.request.future.set_result((raw, timing, error))

    def _admit(self, request: _Request, active: list[_Active]):
        admitted_at = time.perf_counter()
        try:
            state, token, metrics = self.engine.prefill(request)
            row = _Active(request, state, [int(token)], int(metrics["input_tokens"]),
                          float(metrics["seconds"]), admitted_at,
                          peak_allocated=metrics.get("peak_allocated"),
                          peak_reserved=metrics.get("peak_reserved"))
            self._step("prefill", [request.story_id], metrics)
            if token in self.engine.eos_ids or request.max_new_tokens == 1:
                self._finish(row)
            else:
                active.append(row)
        except Exception as exc:
            if not request.future.done():
                request.future.set_result((None, {}, exc))
            if "out of memory" in str(exc).lower():
                with self._lock:
                    self.fatal_error = exc

    def _fail_pending(self, active: list[_Active], exc: Exception):
        with self._lock:
            self.fatal_error = exc
            for row in active:
                if not row.request.future.done():
                    row.request.future.set_result((None, {}, exc))
            active.clear()
            while True:
                try:
                    request = self.queue.get_nowait()
                except Empty:
                    break
                if request is not None and not request.future.done():
                    request.future.set_result((None, {}, exc))

    def _fill(self, active: list[_Active], *, block: bool):
        while len(active) < self.capacity:
            try:
                request = self.queue.get(timeout=self.fill_wait_seconds if not block else None)
            except Empty:
                break
            if request is None:
                self.queue.put(None)
                break
            self._admit(request, active)
            block = False
            if self.fatal_error is not None:
                break

    def _serve(self):
        active: list[_Active] = []
        while True:
            self._fill(active, block=not active)
            if self.fatal_error is not None:
                self._fail_pending(active, self.fatal_error)
                return
            if not active:
                if self._closed:
                    return
                continue
            live_count = len(active)
            try:
                tokens, metrics = self.engine.decode([row.state for row in active], [row.tokens[-1] for row in active])
                if len(tokens) != live_count:
                    raise RuntimeError("continuous decoder returned wrong row count")
            except Exception as exc:
                self._fail_pending(active, exc)
                return
            self._step("decode", [row.request.story_id for row in active], metrics)
            self.metrics["scheduler_seconds"] += float(metrics.get("scheduler_seconds", 0))
            self.metrics["active_slot_steps"] += live_count
            self.metrics["capacity_slot_steps"] += self.capacity
            survivors = []
            for row, token in zip(active, tokens):
                row.tokens.append(int(token))
                row.decode_seconds += float(metrics["seconds"])
                for name in ("peak_allocated", "peak_reserved"):
                    value = metrics.get(name)
                    if value is not None:
                        current = getattr(row, name)
                        setattr(row, name, max(current or 0, value))
                if token in self.engine.eos_ids or len(row.tokens) >= row.request.max_new_tokens:
                    self._finish(row)
                else:
                    survivors.append(row)
            active = survivors


class ContinuousLLM:
    def __init__(self, batcher: ContinuousBatcher, story_id: str):
        self.batcher = batcher
        self.story_id = story_id
        self.device = batcher.engine.device
        self.load_seconds = batcher.engine.load_seconds
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens=1536):
        raw, timing, error = self.batcher.submit(self.story_id, messages, max_new_tokens)
        self.last_generation = timing
        if error is not None:
            raise error
        return raw


class QwenContinuousEngine(ContinuousEngine):
    """Greedy Qwen3 forward steps with a compacted cache for live requests only."""

    def __init__(self, adapter):
        from transformers import DynamicCache

        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = adapter.tokenizer
        self.torch = adapter.torch
        self.device = adapter.device
        self.load_seconds = adapter.load_seconds
        self.DynamicCache = DynamicCache
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self._live_states: list[dict] = []
        self._live_cache = None
        self._live_mask = None
        self.repack_count = 0

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def _begin(self):
        self._sync()
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def _peak(self):
        if self.device != "cuda":
            return None, None
        self._sync()
        return self.torch.cuda.max_memory_allocated(), self.torch.cuda.max_memory_reserved()

    def prefill(self, request: _Request):
        ids = self.tokenizer.apply_chat_template(
            request.messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or not ids:
            raise ValueError("chat template没有产生token")
        if len(ids) + request.max_new_tokens > 4096:
            raise ValueError("输入和输出超过4096 tokens预算，不会截断")
        input_ids = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        mask = self.torch.ones_like(input_ids)
        self._begin()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=mask,
                                past_key_values=self.DynamicCache(), use_cache=True,
                                logits_to_keep=1)
        self._sync()
        seconds = time.perf_counter() - started
        token = int(output.logits[0, -1].argmax().item())
        peak_allocated, peak_reserved = self._peak()
        return {"cache": output.past_key_values, "mask": mask}, token, {
            "input_tokens": len(ids), "padded_input_tokens": 0, "seconds": seconds,
            "peak_allocated": peak_allocated, "peak_reserved": peak_reserved,
        }

    def decode(self, states: list[dict], last_tokens: list[int]):
        if not states or len(states) != len(last_tokens):
            raise ValueError("continuous decode rows mismatch")
        self._begin()
        packing_started = time.perf_counter()
        same_live_rows = (len(states) == len(self._live_states)
                          and all(left is right for left, right in zip(states, self._live_states)))
        if same_live_rows:
            cache = self._live_cache
            attention_mask = self._live_mask
            width = int(attention_mask.shape[-1])
        else:
            self._materialize_live_cache()
            width = max(int(state["mask"].shape[-1]) for state in states)
            lefts = [width - int(state["mask"].shape[-1]) for state in states]
            attention_mask = self.torch.cat([
                self.torch.nn.functional.pad(state["mask"], (left, 0))
                for state, left in zip(states, lefts)
            ], dim=0)
            merged = []
            layer_count = len(states[0]["cache"].layers)
            for layer_index in range(layer_count):
                keys = []
                values = []
                for state, left in zip(states, lefts):
                    layer = state["cache"].layers[layer_index]
                    keys.append(self.torch.nn.functional.pad(layer.keys, (0, 0, left, 0)))
                    values.append(self.torch.nn.functional.pad(layer.values, (0, 0, left, 0)))
                merged.append((self.torch.cat(keys, dim=0), self.torch.cat(values, dim=0)))
            cache = self.DynamicCache(ddp_cache_data=merged)
            self.repack_count += 1
        position_ids = attention_mask.sum(dim=-1, keepdim=True)
        next_mask = self.torch.cat([attention_mask, self.torch.ones((len(states), 1),
                                                dtype=attention_mask.dtype, device=self.device)], dim=-1)
        input_ids = self.torch.tensor(last_tokens, dtype=self.torch.long, device=self.device).view(-1, 1)
        self._sync()
        packing_seconds = time.perf_counter() - packing_started
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=next_mask,
                                position_ids=position_ids,
                                cache_position=self.torch.tensor([width], dtype=self.torch.long, device=self.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
        self._sync()
        seconds = time.perf_counter() - started
        unpacking_started = time.perf_counter()
        tokens = output.logits[:, -1].argmax(dim=-1).tolist()
        for state in states:
            state["mask"] = self.torch.ones((1, state["mask"].shape[-1] + 1),
                                            dtype=attention_mask.dtype, device=self.device)
        self._live_states = list(states)
        self._live_cache = output.past_key_values
        self._live_mask = next_mask
        peak_allocated, peak_reserved = self._peak()
        unpacking_seconds = time.perf_counter() - unpacking_started
        return tokens, {"seconds": seconds, "peak_allocated": peak_allocated,
                        "peak_reserved": peak_reserved,
                        "scheduler_seconds": packing_seconds + unpacking_seconds}

    def text(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def reset_cache(self):
        self._live_states = []
        self._live_cache = None
        self._live_mask = None
        self.repack_count = 0

    def _materialize_live_cache(self):
        if self._live_cache is None:
            return
        width = int(self._live_mask.shape[-1])
        for index, state in enumerate(self._live_states):
            left = width - int(state["mask"].shape[-1])
            layers = [(layer.keys[index:index+1, :, left:, :],
                       layer.values[index:index+1, :, left:, :])
                      for layer in self._live_cache.layers]
            state["cache"] = self.DynamicCache(ddp_cache_data=layers)
        self._live_states = []
        self._live_cache = None
        self._live_mask = None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_continuous_single_story(adapter, text: str) -> bool:
    """Refuse formal runs if the custom forward loop changes capacity-one extraction."""
    original = extract_events(text, device=adapter.device, llm=ExtractionRepairLLM(adapter))
    with ContinuousBatcher(QwenContinuousEngine(adapter), 1) as batcher:
        candidate = extract_events(text, device=adapter.device,
                                   llm=ExtractionRepairLLM(ContinuousLLM(batcher, "SMOKE")))
    fields = ("status", "events", "ignored_spans", "non_event_span_ids", "uncovered_span_ids")
    if any(original[field] != candidate[field] for field in fields):
        raise RuntimeError("Benchmark 3 single-row and continuous capacity-one extraction differ")
    if hasattr(adapter, "clear_traces"):
        adapter.clear_traces()
    return True


def run_continuous_cases(cases: list[dict], engine: ContinuousEngine, capacity: int,
                         output: Path, *, warmup: bool = True, progress=None) -> dict:
    """Run the unchanged extractor; only its token scheduler is replaced."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output.mkdir(parents=True, exist_ok=True)
    row_path = output / "extractor_runs.jsonl"
    step_path = output / "forward_steps.jsonl"
    request_path = output / "request_traces.jsonl"
    rows = _read_jsonl(row_path)
    steps = _read_jsonl(step_path)
    requests = _read_jsonl(request_path)
    expected = [case["id"] for case in cases]
    saved_ids = [row.get("case_id") for row in rows]
    if len(saved_ids) != len(set(saved_ids)) or any(case_id not in expected for case_id in saved_ids):
        raise ValueError("saved continuous rows contain duplicate or unknown stories")
    remaining = [case for case in cases if case["id"] not in set(saved_ids)]
    if not remaining:
        return {"rows": rows, "steps": steps, "requests": requests,
                "wall_seconds": None, "scheduler": None}
    if warmup:
        with ContinuousBatcher(engine, capacity) as batcher:
            extract_events(cases[0]["story"], device=engine.device,
                           llm=ExtractionRepairLLM(ContinuousLLM(batcher, "WARMUP")))
        if batcher.fatal_error is not None:
            raise RuntimeError(f"capacity {capacity} warmup failed: {batcher.fatal_error}") from batcher.fatal_error
        if hasattr(engine, "reset_cache"):
            engine.reset_cache()

    def run_one(case, batcher):
        started = time.perf_counter()
        proxy = ExtractionRepairLLM(ContinuousLLM(batcher, case["id"]))
        try:
            extraction = extract_events(case["story"], device=engine.device, llm=proxy)
            status = extraction["status"]
            error = None
        except Exception as exc:
            extraction = {"status": "error", "events": [], "calls": [], "uncovered_span_ids": []}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        row = {"case_id": case["id"], "system": "benchmark-4-continuous-extractor",
               "status": status,
               "result": {"judge": {"events": extraction["events"], "items": []},
                          "benchmark_stages": {"extraction": extraction}},
               "stage_metrics": {"extraction": {"seconds": time.perf_counter() - started}}}
        if error:
            row["error"] = error
        return row

    started = time.perf_counter()
    persisted_step_records = 0
    persisted_request_records = 0
    with ContinuousBatcher(engine, capacity, step_offset=len(steps),
                           request_offset=len(requests)) as batcher:
        with ThreadPoolExecutor(max_workers=min(len(remaining), max(8, capacity * 2))) as pool:
            futures = {pool.submit(run_one, case, batcher): case["id"] for case in remaining}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                rows.sort(key=lambda value: expected.index(value["case_id"]))
                steps.extend(batcher.steps[persisted_step_records:])
                requests.extend(batcher.records[persisted_request_records:])
                persisted_step_records = len(batcher.steps)
                persisted_request_records = len(batcher.records)
                write_jsonl_atomic(row_path, rows)
                write_jsonl_atomic(step_path, steps)
                write_jsonl_atomic(request_path, requests)
                if progress:
                    progress(len(rows), len(cases), row["case_id"])
    if batcher.fatal_error is not None:
        raise RuntimeError(f"capacity {capacity} failed: {batcher.fatal_error}") from batcher.fatal_error
    return {"rows": rows, "steps": steps, "requests": requests,
            "wall_seconds": time.perf_counter() - started, "scheduler": batcher.metrics}
