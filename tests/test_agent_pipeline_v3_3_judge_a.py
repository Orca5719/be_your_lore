import copy
import json

import pytest

from agent_pipeline_v3_3.judge_a import finalize_answer, judge_frozen_retrieval, validate_answer
from agent_pipeline_v3_3.judge_a_report import classification_metrics
from agent_pipeline_v3_3.judge_a_runner import build_candidate_row, build_manifest
from agent_pipeline_v3_3.cli import build_parser as build_base_parser
from agent_pipeline_v3_3.judge_a_cli import DEFAULT_SOURCE, build_parser, summarize
from agent_pipeline_v3_2.frozen import load_and_validate_source
from agent_pipeline_v3_2.runner import write_jsonl_atomic


EVIDENCE = [{"id": "chunk-a", "text": "雷的左心脏寄宿亚巴顿。"}]


def answer(verdict="contradiction", reason=False):
    value = {
        "evidence_ids": ["L1"],
        "assessment": {"same_subject": True, "evidence_applicable": True,
                       "relation": "direct_conflict", "can_both_be_true": False,
                       "assumptions_required": False},
        "verdict": verdict,
    }
    if reason:
        value["reason"] = "同一心脏的寄宿者互斥。"
    return value


def test_structured_answer_preserves_guard_and_evidence_identity():
    result = finalize_answer(answer(), EVIDENCE, "structured")
    assert result["verdict"] == "contradiction"
    assert result["citations"] == [{"evidence_id": "L1", "chunk_id": "chunk-a"}]
    assert "reason" not in result


def test_short_reason_variant_requires_bounded_reason():
    validate_answer(answer(reason=True), EVIDENCE, "short-reason")
    with pytest.raises(ValueError, match="reason"):
        validate_answer(answer(), EVIDENCE, "short-reason")
    bad = answer(reason=True)
    bad["reason"] = "字" * 81
    with pytest.raises(ValueError, match="reason"):
        validate_answer(bad, EVIDENCE, "short-reason")


@pytest.mark.parametrize("bad", [
    {"evidence_ids": [], "assessment": answer()["assessment"], "verdict": "contradiction"},
    {**answer(), "evidence_ids": ["L2"]},
    {**answer(), "evidence_ids": ["L1", "L1"]},
    {**answer(), "assessment": {**answer()["assessment"], "relation": "unknown"}},
])
def test_invalid_contract_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_answer(bad, EVIDENCE, "structured")


def test_unsupported_decisive_verdict_is_downgraded_to_uncertain():
    value = answer()
    value["assessment"]["can_both_be_true"] = True
    result = finalize_answer(value, EVIDENCE, "structured")
    assert result["verdict"] == "uncertain"
    assert result["model_verdict"] == "contradiction"
    assert result["guard_reason"] == "coexistence_not_ruled_out"


class FakeLLM:
    def __init__(self, value):
        self.value = value
        self.last_batch_generation = {}
        self.calls = []

    def _generate_batch(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        self.last_batch_generation = {"generated_tokens": [40], "truncated_indices": []}
        return [json.dumps(self.value, ensure_ascii=False)]


def retrieval():
    event = {"id": "E1", "actors": ["雷"], "event": "亚巴顿寄宿在右心脏", "modality": "observed"}
    return {"status": "ok", "extraction": {"text": "雷感到右胸异常。"}, "events": [event],
            "items": [{"event_id": "E1", "status": "ok", "event": event,
                       "fact": {"subject": "雷", "normalized_fact": event["event"]},
                       "evidence": EVIDENCE}]}


def test_judge_uses_variant_and_batch_contract():
    llm = FakeLLM(answer(reason=True))
    result = judge_frozen_retrieval(retrieval(), llm, "short-reason", batch_size=8)
    assert result["status"] == "ok"
    assert result["items"][0]["verdict"] == "contradiction"
    assert result["items"][0]["citations"][0]["chunk_id"] == "chunk-a"
    assert llm.calls[0][1] == 320
    assert "assessment" in llm.calls[0][0][0][0]["content"]


def test_classification_metrics_include_uncertain_recall_and_macro_f1():
    matrix = {"consistent": {"consistent": 2},
              "contradiction": {"contradiction": 1, "uncertain": 1},
              "uncertain": {"uncertain": 2, "consistent": 1}}
    result = classification_metrics(matrix)
    assert result["uncertain_recall"] == pytest.approx(2 / 3)
    assert 0 < result["macro_f1"] < 1


def test_candidate_row_preserves_frozen_retrieval_and_replaces_only_judge():
    source = {"case_id": "SL-001", "result": {"benchmark_stages": {"retrieval": retrieval(),
              "judge": {"items": [{"event_id": "E1", "verdict": "uncertain"}]}}}}
    judged = judge_frozen_retrieval(retrieval(), FakeLLM(answer()), "structured")
    row = build_candidate_row(source, judged, [], 0.2, "structured")
    assert row["result"]["benchmark_stages"]["retrieval"] == source["result"]["benchmark_stages"]["retrieval"]
    assert row["result"]["judge"]["items"][0]["verdict"] == "contradiction"
    assert row["stage_metrics"]["extraction"]["seconds"] == 0.0


def test_manifest_binds_both_prompts_and_frozen_source():
    manifest = build_manifest({"story_runs.jsonl": "abc"}, "cuda")
    assert manifest["source_hashes"]["story_runs.jsonl"] == "abc"
    assert set(manifest["prompt_hashes"]) == {"structured", "short-reason"}
    assert manifest["batch_size"] == 8


def test_cli_exposes_judge_a_without_changing_base_commands():
    parser = build_parser()
    assert parser.parse_args(["run", "--device", "cuda"]).command == "run"
    assert build_base_parser().parse_args(["run"]).command == "run"


def test_offline_summary_writes_four_way_table_without_model(tmp_path):
    source = load_and_validate_source(DEFAULT_SOURCE)
    for variant in ("structured", "short-reason"):
        rows = [build_candidate_row(row, row["result"]["benchmark_stages"]["judge"], [], 0.0, variant)
                for row in source["rows"]]
        write_jsonl_atomic(tmp_path / variant / "judge_runs.jsonl", rows)
    result = summarize(tmp_path, source)
    assert result["stories"] == 24
    assert result["facts"] == 80
    assert "Uncertain Recall" in (tmp_path / "judge_a_summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "verdict_changes.csv").exists()
