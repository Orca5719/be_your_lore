from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from agent_pipeline_v2.batch_llm import call_json_batch
from agent_pipeline_v2.model import V2QwenJudge
from agent_pipeline_v3.model import ProfiledV2QwenJudge
from agent_pipeline_v3.profiling import TraceCollector


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, decoded_outputs=None):
        self.decoded_outputs = iter(decoded_outputs) if decoded_outputs is not None else None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=False, return_tensors=None):
        length = int(messages[-1].get("length", 2))
        ids = list(range(10, 10 + length))
        if return_dict:
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, length), dtype=torch.long),
                }
            )
        return ids

    def decode(self, row, skip_special_tokens=True):
        if self.decoded_outputs is not None:
            return next(self.decoded_outputs)
        values = row.tolist() if hasattr(row, "tolist") else row
        return ",".join(str(value) for value in values if value != self.pad_token_id)


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def generate(self, input_ids, attention_mask, streamer=None, stopping_criteria=None, **kwargs):
        if self.fail:
            raise RuntimeError("fake generation failure")
        current = input_ids
        if streamer is not None:
            streamer.put(input_ids)
        for token in (21, 2):
            next_tokens = torch.full((input_ids.shape[0],), token, dtype=torch.long)
            current = torch.cat((current, next_tokens[:, None]), dim=1)
            if streamer is not None:
                streamer.put(next_tokens)
            if stopping_criteria is not None:
                stopping_criteria(current, None)
        if streamer is not None:
            streamer.end()
        return current


def profiled_model(fake_model=None, tokenizer=None):
    value = ProfiledV2QwenJudge.__new__(ProfiledV2QwenJudge)
    value.torch = torch
    value.device = "cpu"
    value.tokenizer = tokenizer or FakeTokenizer()
    value.model = fake_model or FakeModel()
    value.collector = TraceCollector()
    value.last_generation = {}
    value.last_batch_generation = {}
    return value


def plain_model():
    value = V2QwenJudge.__new__(V2QwenJudge)
    value.torch = torch
    value.device = "cpu"
    value.tokenizer = FakeTokenizer()
    value.model = FakeModel()
    value.last_generation = {}
    value.last_batch_generation = {}
    return value


def test_profiled_single_generation_preserves_output_and_records_call():
    llm = profiled_model()
    llm.set_story_id("SL-001")
    raw = llm._generate([{"role": "user", "length": 3}], max_new_tokens=4)

    assert raw == "21,2"
    assert len(llm.collector.records) == 1
    trace = llm.collector.records[0]
    assert trace["story_id"] == "SL-001"
    assert trace["component"] == "extractor"
    assert trace["input_tokens"] == 3
    assert trace["output_tokens"] == 2
    assert trace["batch_size"] == 1
    assert llm.last_generation["call_id"] == trace["call_id"]


def test_profiled_batch_matches_frozen_batch_output_and_records_padding():
    messages = [[{"role": "user", "length": 2}], [{"role": "user", "length": 4}]]
    expected = plain_model()._generate_batch(messages, max_new_tokens=4)
    llm = profiled_model()
    actual = llm._generate_batch(messages, max_new_tokens=4)

    assert actual == expected
    trace = llm.collector.records[0]
    assert trace["component"] == "judge"
    assert trace["batch_size"] == 2
    assert trace["input_tokens"] == 6
    assert trace["padded_input_tokens"] == 2
    assert trace["compute_input_tokens"] == 8
    assert trace["output_tokens"] == 4
    assert trace["prefill_time"] is not None
    assert llm.last_batch_generation["call_id"] == trace["call_id"]


def test_generation_error_is_traced_and_reraised():
    llm = profiled_model(FakeModel(fail=True))
    with pytest.raises(RuntimeError, match="fake generation failure"):
        llm._generate([{"role": "user", "length": 2}], max_new_tokens=4)
    trace = llm.collector.records[0]
    assert trace["status"] == "error"
    assert trace["output_tokens"] == 0
    assert trace["prefill_time"] is None
    assert "fake generation failure" in trace["error"]


def test_clear_traces_drops_warmup_but_keeps_unique_call_ids():
    llm = profiled_model()
    llm._generate([{"role": "user", "length": 2}], max_new_tokens=4)
    first = llm.collector.records[0]["call_id"]
    llm.clear_traces()
    llm._generate_batch([[{"role": "user", "length": 2}]], max_new_tokens=4)
    assert len(llm.collector.records) == 1
    assert llm.collector.records[0]["call_id"] != first


def test_batch_validation_retry_creates_two_traces_and_embeds_call_ids():
    tokenizer = FakeTokenizer(decoded_outputs=["not-json", '{"ok":true}'])
    llm = profiled_model(tokenizer=tokenizer)
    requests = [{"request_id": "E1", "messages": [{"role": "user", "length": 2}]}]
    values, report = call_json_batch(
        llm,
        requests,
        max_output=4,
        validators=[lambda value: None if value.get("ok") is True else (_ for _ in ()).throw(ValueError("bad"))],
    )

    assert values == [{"ok": True}]
    assert report["retry_count"] == 1
    assert len(llm.collector.records) == 2
    assert [call["timing"]["call_id"] for call in report["batch_calls"]] == [
        row["call_id"] for row in llm.collector.records
    ]


def test_cuda_peak_helpers_reset_and_read_per_call_counters():
    calls = []
    fake_cuda = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        reset_peak_memory_stats=lambda: calls.append("reset"),
        max_memory_allocated=lambda: 123,
        max_memory_reserved=lambda: 456,
    )
    llm = ProfiledV2QwenJudge.__new__(ProfiledV2QwenJudge)
    llm.device = "cuda"
    llm.torch = SimpleNamespace(cuda=fake_cuda)

    llm._cuda_begin()
    peaks = llm._cuda_end()

    assert calls == ["sync", "reset", "sync"]
    assert peaks == (123, 456)
