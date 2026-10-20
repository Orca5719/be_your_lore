"""Continuous batching scheduler tests; no model download required."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agent_pipeline_v4.continuous import ContinuousBatcher, ContinuousLLM, ContinuousEngine


class FakeEngine(ContinuousEngine):
    device = "cpu"
    load_seconds = 0.0
    eos_ids = {0}

    def __init__(self):
        self.decode_batches = []
        self.next_token = {"A": [7, 0], "B": [8, 9, 0], "C": [0]}
        self.clock = 0.0

    def prefill(self, request):
        name = request.messages[0]["content"]
        self.clock += 0.1
        return {"name": name}, self.next_token[name].pop(0), {
            "input_tokens": 10, "padded_input_tokens": 0, "seconds": 0.1,
            "peak_allocated": None, "peak_reserved": None,
        }

    def decode(self, states, last_tokens):
        names = [state["name"] for state in states]
        self.decode_batches.append(names)
        self.clock += 0.2
        return [self.next_token[name].pop(0) for name in names], {
            "seconds": 0.2, "peak_allocated": None, "peak_reserved": None,
        }

    def text(self, tokens):
        return ",".join(map(str, tokens))


def test_finished_request_releases_slot_while_other_request_continues():
    engine = FakeEngine()
    barrier = Barrier(2)
    with ContinuousBatcher(engine, capacity=2, fill_wait_seconds=0.05) as batcher:
        def run(name):
            proxy = ContinuousLLM(batcher, name)
            barrier.wait()
            return proxy._generate([{"role": "user", "content": name}], max_new_tokens=5), proxy.last_generation

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(run, ["A", "B"]))
    assert a[0] == "7,0"
    assert b[0] == "8,9,0"
    assert sorted(engine.decode_batches[0]) == ["A", "B"]
    assert engine.decode_batches[1] == ["B"]
    assert batcher.metrics["finished_request_waste_steps"] == 0
    assert batcher.metrics["active_slot_steps"] == 3
    assert batcher.metrics["capacity_slot_steps"] == 4
    assert a[1]["call_id"] != b[1]["call_id"]
    assert a[1]["ttft_ms"] is not None


def test_new_request_enters_vacated_slot_before_survivor_finishes():
    engine = FakeEngine()
    barrier = Barrier(3)
    with ContinuousBatcher(engine, capacity=2, fill_wait_seconds=0.05) as batcher:
        def run(name):
            proxy = ContinuousLLM(batcher, name)
            barrier.wait()
            return proxy._generate([{"role": "user", "content": name}], max_new_tokens=5)

        with ThreadPoolExecutor(max_workers=3) as pool:
            output = dict(zip(["A", "B", "C"], pool.map(run, ["A", "B", "C"])))
    assert output == {"A": "7,0", "B": "8,9,0", "C": "0"}
    assert any(len(batch) == 2 for batch in engine.decode_batches)
    assert batcher.metrics["finished_request_waste_steps"] == 0


def test_truncated_request_raises_only_for_that_request():
    engine = FakeEngine()
    barrier = Barrier(2)
    with ContinuousBatcher(engine, capacity=2, fill_wait_seconds=0.05) as batcher:
        def run(name):
            proxy = ContinuousLLM(batcher, name)
            barrier.wait()
            try:
                return proxy._generate([{"role": "user", "content": name}], max_new_tokens=2)
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(run, ["A", "B"]))
    assert a == "7,0"
    assert isinstance(b, ValueError)
    assert "生成达到上限" in str(b)


def test_backend_failure_releases_all_waiters():
    class BrokenEngine(FakeEngine):
        def decode(self, states, last_tokens):
            raise RuntimeError("CUDA out of memory")

    engine = BrokenEngine()
    barrier = Barrier(2)
    with ContinuousBatcher(engine, capacity=2, fill_wait_seconds=0.05) as batcher:
        def run(name):
            proxy = ContinuousLLM(batcher, name)
            barrier.wait()
            with pytest.raises(RuntimeError, match="out of memory"):
                proxy._generate([{"role": "user", "content": name}], max_new_tokens=5)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, ["A", "B"]))
    assert batcher.fatal_error is not None


def test_empty_output_eos_and_zero_decode_time_are_valid():
    engine = FakeEngine()
    with ContinuousBatcher(engine, capacity=1) as batcher:
        proxy = ContinuousLLM(batcher, "C")
        assert proxy._generate([{"role": "user", "content": "C"}], max_new_tokens=3) == "0"
    assert proxy.last_generation["generated_tokens"] == 1
    assert proxy.last_generation["decode_seconds"] == 0


def test_qwen_cache_rebatch_preserves_each_requests_greedy_tokens():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from agent_pipeline_v4.continuous import QwenContinuousEngine

    torch.manual_seed(3)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=128, eos_token_id=63,
                         pad_token_id=0)
    model = Qwen3ForCausalLM(config).eval()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 63

        def apply_chat_template(self, messages, **_):
            return [int(item) for item in messages[0]["content"].split(",")]

        def decode(self, tokens, skip_special_tokens=True):
            return ",".join(str(token) for token in tokens if token not in (0, 63))

    adapter = type("Adapter", (), {"model": model, "tokenizer": Tokenizer(),
                                    "torch": torch, "device": "cpu", "load_seconds": 0.0})()
    engine = QwenContinuousEngine(adapter)
    prompts = [[4, 5, 6], [7, 8, 9, 10], [11, 12]]
    states = []
    tokens = []
    for index in (0, 1):
        request = _make_request(",".join(map(str, prompts[index])))
        state, first, _ = engine.prefill(request)
        states.append(state)
        tokens.append([first])
    second, _ = engine.decode(states, [value[-1] for value in tokens])
    for index, token in enumerate(second):
        tokens[index].append(token)
    third, _ = engine.decode(states, [value[-1] for value in tokens])
    for index, token in enumerate(third):
        tokens[index].append(token)
    assert engine.repack_count == 1
    state_c, first_c, _ = engine.prefill(_make_request("11,12"))
    mixed, _ = engine.decode([states[1], state_c], [tokens[1][-1], first_c])
    tokens[1].append(mixed[0])
    generated_c = [first_c, mixed[1]]
    assert engine.repack_count == 2
    with torch.inference_mode():
        expected = [model.generate(torch.tensor([prompt]), do_sample=False,
                                   max_new_tokens=length, pad_token_id=0)[0, len(prompt):].tolist()
                    for prompt, length in zip(prompts, (3, 4, 2))]
    assert tokens[0] == expected[0]
    assert tokens[1] == expected[1]
    assert generated_c == expected[2]
    engine.reset_cache()
    assert engine.repack_count == 0


def _make_request(content):
    from concurrent.futures import Future
    from agent_pipeline_v4.continuous import _Request
    return _Request("fixture", [{"role": "user", "content": content}], 5, Future())


def test_runner_persists_stories_and_forward_steps_for_resume(tmp_path):
    import json
    from agent_pipeline_v4.continuous import run_continuous_cases

    class JsonEngine(FakeEngine):
        eos_ids = set(range(1, 100))
        def __init__(self):
            super().__init__()
            self.payloads = {}
            self.serial = 0

        def prefill(self, request):
            payload = json.loads(request.messages[1]["content"])
            self.serial += 1
            self.payloads[self.serial] = json.dumps({
                "events": [], "ignored_spans": [],
                "non_event_span_ids": list(payload["target_spans"]),
            }, ensure_ascii=False)
            return {"name": request.story_id}, self.serial, {
                "input_tokens": 10, "padded_input_tokens": 0, "seconds": 0.1,
                "peak_allocated": None, "peak_reserved": None,
            }

        def text(self, tokens):
            return self.payloads[tokens[0]]

    engine = JsonEngine()
    cases = [{"id": "A", "story": "雷走入房间。"},
             {"id": "B", "story": "月城抬起手。"}]
    result = run_continuous_cases(cases, engine, 2, tmp_path, warmup=False)
    assert [row["case_id"] for row in result["rows"]] == ["A", "B"]
    assert all(row["status"] == "ok" for row in result["rows"])
    assert len(result["steps"]) == 2
    assert len(result["requests"]) == 2
    resumed = run_continuous_cases(cases, engine, 2, tmp_path, warmup=False)
    assert resumed["rows"] == result["rows"]
    assert resumed["wall_seconds"] is None
    assert len(resumed["steps"]) == 2


def test_original_extractor_format_retry_and_coverage_recovery_survive_scheduler(tmp_path):
    import json
    from agent_pipeline_v4.continuous import run_continuous_cases

    class RepairEngine(FakeEngine):
        eos_ids = set(range(1, 100))

        def __init__(self):
            super().__init__()
            self.serial = 0
            self.output = {}

        def prefill(self, request):
            self.serial += 1
            payload = json.loads(request.messages[1]["content"])
            target = list(payload["target_spans"])
            if self.serial == 1:
                raw = "not-json"
            elif self.serial == 2:
                raw = json.dumps({"events": [], "ignored_spans": [], "non_event_span_ids": []})
            else:
                raw = json.dumps({"events": [], "ignored_spans": [], "non_event_span_ids": target})
            self.output[self.serial] = raw
            return {}, self.serial, {"input_tokens": 10, "seconds": 0.1,
                                     "peak_allocated": None, "peak_reserved": None}

        def text(self, tokens):
            return self.output[tokens[0]]

    engine = RepairEngine()
    result = run_continuous_cases([{"id": "A", "story": "雷走入房间。"}], engine, 1,
                                  tmp_path, warmup=False)
    extraction = result["rows"][0]["result"]["benchmark_stages"]["extraction"]
    assert extraction["status"] == "ok"
    attempts = extraction["calls"][0]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["error", "error", "ok"]
    assert attempts[-1]["purpose"] == "recover_missing_targets"
    assert len(result["requests"]) == 3
