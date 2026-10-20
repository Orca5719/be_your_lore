import copy
import json

import pytest

from agent_pipeline_v3_3.judge_a2 import finalize_answer, judge_frozen_retrieval, validate_answer
from agent_pipeline_v3_3.judge_a import _messages as first_round_messages
from agent_pipeline_v3_3.judge_a2 import _messages as second_round_messages
from agent_pipeline_v3_3.judge_a2_cli import DEFAULT_SOURCE, build_parser, build_manifest, summarize, resolve_a1_result
from agent_pipeline_v3_3.judge_a_runner import build_candidate_row
from agent_pipeline_v3_2.frozen import load_and_validate_source
from agent_pipeline_v3_2.runner import write_jsonl_atomic
from agent_pipeline_v2_1.oracle_benchmark import build_messages as baseline_messages


EVIDENCE = [{"id": "chunk-a", "text": "雷的左心脏寄宿亚巴顿。"}]


def answer(variant, reason=None):
    value = {"evidence_ids": ["L1"],
             "assessment": {"same_subject": True, "evidence_applicable": True,
                            "relation": "direct_conflict", "can_both_be_true": False,
                            "assumptions": []},
             "verdict": "contradiction"}
    if variant != "assumptions-list":
        value["reason"] = reason or "同一心脏不能同时由不同天使寄宿。"
    return value


@pytest.mark.parametrize("variant", ["assumptions-list", "rationale-120", "rationale-240"])
def test_old_assessment_contract_and_evidence_identity(variant):
    value = answer(variant)
    validate_answer(value, EVIDENCE, variant)
    result = finalize_answer(value, EVIDENCE, variant)
    assert result["verdict"] == "contradiction"
    assert result["citations"] == [{"evidence_id": "L1", "chunk_id": "chunk-a"}]
    assert result["assessment"]["assumptions"] == []


def test_missing_premise_downgrades_decisive_verdict():
    value = answer("rationale-120")
    value["assessment"]["assumptions"] = ["必须假定两个情节发生在同一天"]
    result = finalize_answer(value, EVIDENCE, "rationale-120")
    assert result["verdict"] == "uncertain"
    assert result["model_verdict"] == "contradiction"


@pytest.mark.parametrize("variant,limit", [("rationale-120", 120), ("rationale-240", 240)])
def test_reason_lengths_are_distinct(variant, limit):
    validate_answer(answer(variant, "字" * limit), EVIDENCE, variant)
    with pytest.raises(ValueError, match="reason"):
        validate_answer(answer(variant, "字" * (limit + 1)), EVIDENCE, variant)


def test_list_only_variant_rejects_reason_and_boolean_assumption():
    value = answer("assumptions-list")
    value["reason"] = "多余字段"
    with pytest.raises(ValueError):
        validate_answer(value, EVIDENCE, "assumptions-list")
    value.pop("reason")
    value["assessment"]["assumptions"] = False
    with pytest.raises(ValueError):
        validate_answer(value, EVIDENCE, "assumptions-list")


class FakeLLM:
    def __init__(self, output):
        self.output = output
        self.calls = []
        self.last_batch_generation = {}

    def _generate_batch(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        self.last_batch_generation = {"generated_tokens": [50], "truncated_indices": []}
        return [json.dumps(self.output, ensure_ascii=False)]


def retrieval():
    event = {"id": "E1", "actors": ["雷"], "event": "亚巴顿寄宿在右心脏", "modality": "observed"}
    return {"status": "ok", "extraction": {"text": "雷感到右胸异常。"}, "events": [event],
            "items": [{"event_id": "E1", "status": "ok", "event": event,
                       "fact": {"subject": "雷", "normalized_fact": event["event"], "dimension": "physical_rule"},
                       "evidence": EVIDENCE}]}


@pytest.mark.parametrize("variant,budget", [("assumptions-list", 320), ("rationale-120", 448), ("rationale-240", 640)])
def test_each_variant_runs_with_its_own_budget(variant, budget):
    llm = FakeLLM(answer(variant))
    result = judge_frozen_retrieval(retrieval(), llm, variant, batch_size=8)
    assert result["status"] == "ok"
    assert llm.calls[0][1] == budget
    assert result["items"][0]["verdict"] == "contradiction"


def test_a2_cli_and_manifest_lock_three_prompts():
    assert build_parser().parse_args(["run", "--device", "cuda"]).command == "run"
    assert build_parser().parse_args(["run", "--a1-result", "C:/runs/a1"]).a1_result.name == "a1"
    manifest = build_manifest({"story_runs.jsonl": "abc"}, "cuda")
    assert set(manifest["prompt_hashes"]) == {"assumptions-list", "rationale-120", "rationale-240"}
    assert manifest["batch_size"] == 8


def test_a1_result_accepts_directory_or_markdown_path(tmp_path):
    expected = tmp_path / "judge_a_summary.json"
    assert resolve_a1_result(tmp_path) == expected
    assert resolve_a1_result(tmp_path / "judge_a_summary.md") == expected


def test_all_a_variants_preserve_full_judge_user_payload():
    source = retrieval()["items"][0]
    frozen = retrieval()
    wire = {"fixture_id": source["event_id"], "fact": source["fact"],
            "story_context": frozen["extraction"]["text"], "lore": source["evidence"]}
    expected = baseline_messages(wire, "v2.1")[1]["content"]
    for variant in ("structured", "short-reason"):
        assert first_round_messages(frozen, source, variant)[1]["content"] == expected
    for variant in ("assumptions-list", "rationale-120", "rationale-240"):
        assert second_round_messages(frozen, source, variant)[1]["content"] == expected


def test_offline_seven_way_report_does_not_load_model(tmp_path):
    source = load_and_validate_source(DEFAULT_SOURCE)
    for variant in ("assumptions-list", "rationale-120", "rationale-240"):
        rows = [build_candidate_row(row, row["result"]["benchmark_stages"]["judge"], [], 0.0, variant)
                for row in source["rows"]]
        write_jsonl_atomic(tmp_path / variant / "judge_runs.jsonl", rows)
    summary = summarize(tmp_path, source)
    assert summary["stories"] == 24
    assert set(summary["variants"]) == {"assumptions-list", "rationale-120", "rationale-240"}
    markdown = (tmp_path / "judge_a2_summary.md").read_text(encoding="utf-8")
    assert "Uncertain Recall" in markdown
    assert "Full B3" in markdown
