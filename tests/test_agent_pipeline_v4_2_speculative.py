"""Focused checks for the batch-one prompt-lookup adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_pipeline_v4_2.speculative import _ensure_manifest, apply_review_corrections
from agent_pipeline_v4_2.speculative_model import CommitStreamer, prompt_lookup_generation


class FakeTensor:
    def __init__(self, n: int):
        self.shape = (1, n)
        self.n = n

    def numel(self):
        return self.n


class FakeModel:
    device = "cpu"

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        streamer = kwargs["streamer"]
        streamer.put(FakeTensor(7))  # prompt
        streamer.put(FakeTensor(3))
        streamer.put(FakeTensor(1))
        streamer.end()
        if self.fail:
            raise RuntimeError("generation failed")
        return "generated"


def test_streamer_counts_committed_chunks_and_first_token():
    times = iter([1.0])
    streamer = CommitStreamer(clock=lambda: next(times))
    streamer.put(FakeTensor(100))
    assert streamer.first_commit_at is None
    streamer.put(FakeTensor(3))
    streamer.put(FakeTensor(1))
    assert streamer.first_commit_at == 1.0
    assert streamer.chunk_sizes == [3, 1]


def test_prompt_lookup_injects_only_generation_option_and_restores():
    model = FakeModel()
    original = model.generate
    observed = {}
    with prompt_lookup_generation(model, 5, observed):
        assert model.generate(input_ids=FakeTensor(7), stopping_criteria="timer") == "generated"
    assert model.generate == original
    call = model.calls[0]
    assert call["prompt_lookup_num_tokens"] == 5
    assert "stopping_criteria" not in call
    assert observed["streamer"].chunk_sizes == [3, 1]


def test_prompt_lookup_restores_generate_after_error():
    model = FakeModel(fail=True)
    original = model.generate
    with pytest.raises(RuntimeError, match="generation failed"):
        with prompt_lookup_generation(model, 5, {}):
            model.generate(input_ids=FakeTensor(7))
    assert model.generate == original


def test_prompt_lookup_rejects_batch_and_invalid_token_count():
    model = FakeModel()
    with pytest.raises(ValueError, match="batch size 1"):
        with prompt_lookup_generation(model, 5, {}):
            model.generate(input_ids=type("Batch", (), {"shape": (2, 7)})())
    with pytest.raises(ValueError, match="positive integer"):
        with prompt_lookup_generation(model, 0, {}):
            pass


def test_manifest_refuses_changed_baseline_or_generation_config(tmp_path: Path):
    identity = {"schema_version": "test", "baseline_files": {"rows": "abc"}, "tokens": 5}
    _ensure_manifest(tmp_path, identity)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == identity
    _ensure_manifest(tmp_path, dict(identity))
    with pytest.raises(ValueError, match="变化"):
        _ensure_manifest(tmp_path, {**identity, "tokens": 4})
    with pytest.raises(ValueError, match="变化"):
        _ensure_manifest(tmp_path, {**identity, "baseline_files": {"rows": "changed"}})


def test_review_correction_checks_signature_and_preserves_original():
    old = {"cases": {"SL-1": {"E1": {"signature": "abc", "label": "valid_checkable",
                                           "gold_ids": ["G1"], "note": None}}}}
    correction = {"schema_version": "benchmark-4.2B-baseline-review-corrections-v1",
                  "corrections": [{"case_id": "SL-1", "event_id": "E1", "signature": "abc",
                                   "from_label": "valid_checkable", "from_gold_ids": ["G1"],
                                   "to_label": "hallucinated", "to_gold_ids": [], "reason": "unsupported"}]}
    fixed = apply_review_corrections(old, correction)
    assert fixed["cases"]["SL-1"]["E1"]["label"] == "hallucinated"
    assert old["cases"]["SL-1"]["E1"]["label"] == "valid_checkable"
    correction["corrections"][0]["signature"] = "stale"
    with pytest.raises(ValueError, match="不匹配"):
        apply_review_corrections(old, correction)
