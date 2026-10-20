"""Benchmark 4A fixed-batch extraction without loading the real model."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import json

from agent_pipeline_v4.static import StaticBatcher, QueuedLLM, run_static_cases, compare_single_story


class FakeBatchModel:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self):
        self.calls = []
        self.last_batch_generation = {}

    def _generate_batch(self, messages, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        index = len(self.calls)
        self.last_batch_generation = {
            "call_id": f"LLM-{index:06d}",
            "input_tokens": [10] * len(messages),
            "padded_input_tokens": 0,
            "generated_tokens": [2] * len(messages),
            "truncated_indices": [],
            "seconds": 1.0,
            "profile": {"call_id": f"LLM-{index:06d}", "component": "judge", "batch_size": len(messages), "total_time": 1.0, "prefill_time": 0.2, "decode_time": 0.8, "ttft_ms": 200.0, "input_tokens": 10 * len(messages), "padded_input_tokens": 0, "output_tokens": 2 * len(messages), "peak_allocated": None, "peak_reserved": None},
        }
        return [row[0]["content"] for row in messages]


def test_collects_requests_from_different_stories_into_one_batch():
    model = FakeBatchModel()
    barrier = Barrier(4)
    with StaticBatcher(model, batch_size=4, fill_wait_seconds=0.2) as batcher:
        def work(story_id):
            proxy = QueuedLLM(batcher, story_id)
            barrier.wait()
            result = proxy._generate([{"role": "user", "content": story_id}], max_new_tokens=1536)
            return result, proxy.last_generation

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(work, ("A", "B", "C", "D")))
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 4
    assert [row[0] for row in results] == ["A", "B", "C", "D"]
    assert all(row[1]["call_id"] == "LLM-000001" for row in results)
    assert batcher.records[0]["component"] == "extractor"
    assert sorted(batcher.records[0]["story_ids"]) == ["A", "B", "C", "D"]


def test_truncation_is_reported_only_to_affected_row():
    model = FakeBatchModel()
    original = model._generate_batch

    def generate(messages, max_new_tokens):
        rows = original(messages, max_new_tokens)
        model.last_batch_generation["truncated_indices"] = [1]
        return rows

    model._generate_batch = generate
    barrier = Barrier(2)
    with StaticBatcher(model, batch_size=2, fill_wait_seconds=0.2) as batcher:
        def work(story_id):
            proxy = QueuedLLM(batcher, story_id)
            barrier.wait()
            try:
                return proxy._generate([{"role": "user", "content": story_id}])
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(work, ("A", "B")))
    assert sum(isinstance(value, ValueError) for value in results) == 1
    assert sum(isinstance(value, str) for value in results) == 1


def test_runner_keeps_story_rows_separate_and_reuses_frozen_extractor(tmp_path):
    class ExtractorModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            super()._generate_batch(messages, max_new_tokens)
            outputs = []
            for row in messages:
                payload = json.loads(row[1]["content"])
                outputs.append(json.dumps({
                    "events": [], "ignored_spans": [],
                    "non_event_span_ids": list(payload["target_spans"]),
                }, ensure_ascii=False))
            return outputs

    model = ExtractorModel()
    cases = [{"id": "A", "story": "雷走入房间。"}, {"id": "B", "story": "月城抬起手。"}]
    result = run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert [row["case_id"] for row in result["rows"]] == ["A", "B"]
    assert all(row["status"] == "ok" for row in result["rows"])
    assert all(row["result"]["benchmark_stages"]["extraction"]["events"] == [] for row in result["rows"])
    assert len(result["calls"]) == 1
    assert result["calls"][0]["batch_size"] == 2
    resumed = run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert resumed["rows"] == result["rows"]
    assert resumed["wall_seconds"] is None


def test_original_extractor_retry_is_batched_without_losing_attempts(tmp_path):
    class RetryModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            super()._generate_batch(messages, max_new_tokens)
            if len(self.calls) == 1:
                return ["not json"] * len(messages)
            outputs = []
            for row in messages:
                payload = json.loads(row[1]["content"])
                outputs.append(json.dumps({"events": [], "ignored_spans": [],
                                           "non_event_span_ids": list(payload["target_spans"])}, ensure_ascii=False))
            return outputs

    model = RetryModel()
    cases = [{"id": "A", "story": "雷走入房间。"}, {"id": "B", "story": "月城抬起手。"}]
    result = run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert [call["batch_size"] for call in result["calls"]] == [2, 2]
    assert all([attempt["status"] for attempt in row["result"]["benchmark_stages"]["extraction"]["calls"][0]["attempts"]] == ["error", "ok"] for row in result["rows"])


def test_cuda_oom_is_not_silently_retried_at_lower_batch_size(tmp_path):
    class OomModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            self.calls.append((messages, max_new_tokens))
            raise RuntimeError("CUDA out of memory")

    import pytest
    model = OomModel()
    cases = [{"id": str(index), "story": f"人物{index}走入房间。"} for index in range(4)]
    with pytest.raises(RuntimeError, match="out of memory"):
        run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert len(model.calls) == 1
    assert len(model.calls[0][0]) == 2


def test_warmup_oom_stops_before_formal_stories(tmp_path):
    class OomModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            self.calls.append((messages, max_new_tokens))
            raise RuntimeError("CUDA out of memory")

    import pytest
    model = OomModel()
    with pytest.raises(RuntimeError, match="out of memory"):
        run_static_cases([{"id": "A", "story": "雷走入房间。"}], model, 8, tmp_path, warmup=True)
    assert len(model.calls) == 1
    assert not (tmp_path / "extractor_runs.jsonl").exists()


def test_smoke_compares_batch_one_with_original_generation():
    class ParityModel(FakeBatchModel):
        def _output(self, messages):
            payload = json.loads(messages[1]["content"])
            return json.dumps({"events": [], "ignored_spans": [],
                               "non_event_span_ids": list(payload["target_spans"])}, ensure_ascii=False)

        def _generate(self, messages, max_new_tokens):
            self.last_generation = {}
            return self._output(messages)

        def _generate_batch(self, messages, max_new_tokens):
            super()._generate_batch(messages, max_new_tokens)
            return [self._output(row) for row in messages]

    assert compare_single_story(ParityModel(), "雷走入房间。") is True


def test_malformed_batch_metadata_releases_every_waiting_story():
    class BrokenModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            rows = super()._generate_batch(messages, max_new_tokens)
            self.last_batch_generation["generated_tokens"] = [2]
            return rows

    model = BrokenModel()
    barrier = Barrier(2)
    with StaticBatcher(model, 2, fill_wait_seconds=0.2) as batcher:
        def work(story_id):
            proxy = QueuedLLM(batcher, story_id)
            barrier.wait()
            try:
                proxy._generate([{"role": "user", "content": story_id}])
            except RuntimeError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            errors = list(pool.map(work, ("A", "B")))
    assert errors == ["batch model omitted row-level token counts"] * 2


def test_failed_stories_are_retried_on_resume_without_duplicate_rows(tmp_path):
    class RecoveringModel(FakeBatchModel):
        oom = True

        def _generate_batch(self, messages, max_new_tokens):
            if self.oom:
                self.calls.append((messages, max_new_tokens))
                raise RuntimeError("CUDA out of memory")
            super()._generate_batch(messages, max_new_tokens)
            return [json.dumps({"events": [], "ignored_spans": [],
                                "non_event_span_ids": list(json.loads(row[1]["content"])["target_spans"])})
                    for row in messages]

    import pytest
    model = RecoveringModel()
    cases = [{"id": "A", "story": "雷走入房间。"}, {"id": "B", "story": "月城抬起手。"}]
    with pytest.raises(RuntimeError, match="out of memory"):
        run_static_cases(cases, model, 2, tmp_path, warmup=False)
    model.oom = False
    recovered = run_static_cases(cases, model, 2, tmp_path, warmup=False)
    assert [row["case_id"] for row in recovered["rows"]] == ["A", "B"]
    assert all(row["status"] == "ok" for row in recovered["rows"])


def test_pre_generation_failure_does_not_reuse_previous_trace():
    class PreflightFailureModel(FakeBatchModel):
        def _generate_batch(self, messages, max_new_tokens):
            if messages[0][0]["content"] == "bad":
                raise ValueError("input budget exceeded")
            return super()._generate_batch(messages, max_new_tokens)

    model = PreflightFailureModel()
    with StaticBatcher(model, 1) as batcher:
        good = QueuedLLM(batcher, "A")
        assert good._generate([{"role": "user", "content": "good"}]) == "good"
        bad = QueuedLLM(batcher, "B")
        try:
            bad._generate([{"role": "user", "content": "bad"}])
        except ValueError as exc:
            assert str(exc) == "input budget exceeded"
        else:
            raise AssertionError("expected pre-generation failure")
        assert bad.last_generation == {}
    assert len(batcher.records) == 1
