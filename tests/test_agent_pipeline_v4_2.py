"""Focused tests for Benchmark 4.2A."""

import pytest

from agent_pipeline_v4_2.model import TorchCompatibleProcessor, constrained_generation, extractor_json_schema
from agent_pipeline_v4_2.quality import build_review_template
from agent_pipeline_v2_2.review import _event_signature as frozen_event_signature


def test_schema_preserves_frozen_extractor_fields_and_nullable_mental_state():
    schema = extractor_json_schema()
    assert set(schema["properties"]) == {"events", "ignored_spans", "non_event_span_ids"}
    event = schema["properties"]["events"]["items"]
    assert set(event["required"]) == {
        "actors", "event", "mental_state", "explicit", "modality",
        "conditions", "source_ids", "context_ids", "check_reason",
    }
    assert set(event["properties"]["mental_state"]["type"]) == {"string", "null"}


def test_constraint_is_injected_for_one_call_then_removed():
    class Model:
        def generate(self, **kwargs):
            return kwargs

    model = Model()
    before = model.generate
    with constrained_generation(model, lambda: "processor"):
        assert model.generate(max_new_tokens=8)["logits_processor"] == ["processor"]
    assert model.generate(max_new_tokens=8) == {"max_new_tokens": 8}
    assert model.generate == before


def test_constraint_is_removed_after_generation_error():
    class Model:
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    model = Model()
    before = model.generate
    with pytest.raises(RuntimeError, match="generation failed"):
        with constrained_generation(model, lambda: "processor"):
            model.generate()
    assert model.generate == before


def test_torch_compatible_processor_returns_original_device_and_dtype():
    import torch

    class CPUOnlyProcessor:
        def __call__(self, input_ids, scores):
            assert scores.device.type == "cpu"
            scores[0, 0] = float("-inf")
            return scores

    scores = torch.zeros((1, 4), device="cpu", dtype=torch.float16)
    output = TorchCompatibleProcessor(CPUOnlyProcessor())(torch.ones((1, 2), dtype=torch.long), scores)
    assert output.device.type == "cpu"
    assert output.dtype == scores.dtype
    assert torch.isneginf(output[0, 0])


def test_review_template_never_reuses_label_by_event_id_alone():
    old = {"case_id": "S1", "result": {"benchmark_stages": {"extraction": {"events": [
        {"id": "E1", "actors": ["雷"], "event": "雷左胸疼", "mental_state": None,
         "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"],
         "context_ids": [], "check_reason": "state_time_space"},
    ]}}}}
    new = {"case_id": "S1", "result": {"benchmark_stages": {"extraction": {"events": [
        {**old["result"]["benchmark_stages"]["extraction"]["events"][0], "event": "雷右胸疼"},
    ]}}}}
    old_review = {"system_event_labels": {"candidate:S1": {"E1": {"label": "valid_checkable", "gold_ids": ["G1"], "note": "old"}}}}
    template = build_review_template([old], [new], old_review, "constrained")
    assert template["cases"]["S1"]["E1"]["label"] is None
    assert template["cases"]["S1"]["E1"]["gold_ids"] == []
    assert template["pending"] == ["S1/E1"]


def test_review_template_rejects_stale_frozen_label_signature():
    event = {"id": "E1", "actors": ["雷"], "event": "雷左胸疼", "mental_state": None,
             "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"],
             "context_ids": [], "check_reason": "state_time_space"}
    row = {"case_id": "S1", "result": {"benchmark_stages": {"extraction": {"events": [event]}}}}
    old_review = {"system_event_labels": {"candidate:S1": {"E1": {
        "signature": "stale", "label": "valid_checkable", "gold_ids": ["G1"]}}}}
    template = build_review_template([row], [row], old_review, "baseline")
    assert template["cases"]["S1"]["E1"]["label"] is None


def test_review_template_reuses_exact_frozen_event_with_valid_signature():
    event = {"id": "E1", "actors": ["雷"], "event": "雷左胸疼", "mental_state": None,
             "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"],
             "context_ids": [], "check_reason": "state_time_space"}
    row = {"case_id": "S1", "result": {"benchmark_stages": {"extraction": {"events": [event]}}}}
    signature = frozen_event_signature("S1", "candidate", event)
    old_review = {"system_event_labels": {"candidate:S1": {"E1": {
        "signature": signature, "label": "valid_checkable", "gold_ids": ["G1"]}}}}
    template = build_review_template([row], [row], old_review, "baseline")
    assert template["pending"] == []
    assert template["cases"]["S1"]["E1"]["gold_ids"] == ["G1"]
