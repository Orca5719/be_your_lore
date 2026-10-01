from __future__ import annotations

import copy

import pytest

from agent_pipeline_v3_2.lean_judge import (
    finalize_lean_answer,
    judge_frozen_retrieval,
    validate_lean_answer,
)


def evidence():
    return [
        {"id": "chunk-a", "text": "雷拥有两颗心脏。"},
        {"id": "chunk-b", "text": "亚巴顿寄宿在左心脏。"},
    ]


def test_validate_lean_answer_accepts_minimal_contract():
    validate_lean_answer({"verdict": "consistent", "evidence_ids": ["L1"]}, evidence())
    validate_lean_answer({"verdict": "uncertain", "evidence_ids": []}, evidence())


@pytest.mark.parametrize(
    "value",
    [
        {"verdict": "consistent", "evidence_ids": []},
        {"verdict": "contradiction", "evidence_ids": ["L9"]},
        {"verdict": "uncertain", "evidence_ids": ["L1", "L1"]},
        {"verdict": "maybe", "evidence_ids": []},
        {"verdict": "uncertain", "evidence_ids": [], "reason": "extra"},
    ],
)
def test_validate_lean_answer_rejects_invalid_contract(value):
    with pytest.raises(ValueError):
        validate_lean_answer(value, evidence())


def test_finalize_resolves_aliases_without_inventing_reason():
    result = finalize_lean_answer(
        {"verdict": "contradiction", "evidence_ids": ["L2"]}, evidence()
    )
    assert result == {
        "verdict": "contradiction",
        "evidence_ids": ["L2"],
        "citations": [{"evidence_id": "L2", "chunk_id": "chunk-b"}],
    }
    assert "reason" not in result
    assert "assessment" not in result


class FakeLLM:
    device = "cuda"
    load_seconds = 0.0

    def __init__(self):
        self.last_batch_generation = {}
        self.calls = []

    def _generate_batch(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        self.last_batch_generation = {
            "call_id": "LLM-000001",
            "seconds": 0.1,
            "input_tokens": [10],
            "useful_input_tokens": 10,
            "padded_input_tokens": 0,
            "generated_tokens": [8],
            "total_generated_tokens": 8,
            "truncated_indices": [],
        }
        return ['{"verdict":"consistent","evidence_ids":["L1"]}']


def test_judge_frozen_retrieval_uses_128_token_budget_and_keeps_evidence():
    retrieval = {
        "status": "ok",
        "extraction": {"text": "雷拥有两颗心脏。"},
        "events": [{"id": "E1"}],
        "items": [{
            "event_id": "E1",
            "status": "ok",
            "event": {"id": "E1", "actors": ["雷"], "event": "雷拥有两颗心脏", "modality": "observed"},
            "fact": {"subject": "雷", "normalized_fact": "雷拥有两颗心脏"},
            "evidence": evidence(),
        }],
    }
    llm = FakeLLM()
    result = judge_frozen_retrieval(retrieval, llm, batch_size=8)
    assert result["status"] == "ok"
    assert result["items"][0]["verdict"] == "consistent"
    assert result["items"][0]["citations"][0]["chunk_id"] == "chunk-a"
    assert "reason" not in result["items"][0]
    assert llm.calls[0][1] == 128
