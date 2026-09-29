import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fixture_item(expected="contradiction"):
    return {
        "fixture_id": "OR-X-G1",
        "case_id": "X",
        "gold_fact_id": "G1",
        "expected_verdict": expected,
        "fact": {
            "subject": "雷",
            "predicate": "寄宿",
            "object": "右侧心脏中的亚巴顿",
            "normalized_fact": "亚巴顿寄宿在雷的右侧心脏",
            "type": "character_attribute",
            "dimension": "physical_rule",
            "source_anchors": [],
            "context_anchors": [],
        },
        "story_context": "雷的右侧心脏寄宿亚巴顿。",
        "lore": [{"id": "L-raw", "text": "亚巴顿寄宿在雷的左侧心脏里。", "heading_path": ["亚巴顿"]}],
        "oracle_selection": {"source": "minimum_evidence_set", "rationale": "gold"},
    }


def test_oracle_messages_use_the_selected_prompt_and_gold_lore_only():
    from agent_pipeline_v2_1.oracle_benchmark import build_messages

    messages = build_messages(fixture_item(), "v2.1")
    payload = json.loads(messages[1]["content"])

    assert "无法同时为真" in messages[0]["content"]
    assert payload["event"]["event"] == "亚巴顿寄宿在雷的右侧心脏"
    assert payload["lore"] == [{"evidence_id": "L1", "text": "亚巴顿寄宿在雷的左侧心脏里。", "heading_path": ["亚巴顿"]}]
    assert "expected_verdict" not in payload
    assert '"evidence_id":"L1"' in messages[0]["content"]
    assert "不得把citations写成字符串数组" in messages[0]["content"]


def test_oracle_messages_can_replay_frozen_v1_prompt():
    from agent_pipeline_v2_1.oracle_benchmark import build_messages

    messages = build_messages(fixture_item(), "v1")
    frozen = (ROOT / "agent_pipeline_v2" / "prompts" / "judge_v1.txt").read_text(encoding="utf-8")
    assert messages[0]["content"] == frozen


def test_scoring_reports_accuracy_macro_f1_and_contradiction_fp_fn():
    from agent_pipeline_v2_1.oracle_benchmark import score_oracle_rows

    rows = [
        {"expected_verdict": "consistent", "predicted_verdict": "consistent", "status": "ok"},
        {"expected_verdict": "contradiction", "predicted_verdict": "uncertain", "status": "ok"},
        {"expected_verdict": "uncertain", "predicted_verdict": "contradiction", "status": "ok", "unsupported_inference": True},
    ]
    metrics = score_oracle_rows(rows)

    assert metrics["accuracy"] == 1 / 3
    assert metrics["contradiction_false_positive"] == 1
    assert metrics["contradiction_false_negative"] == 1
    assert metrics["unsupported_inference_count"] == 1
    assert metrics["per_class"]["consistent"]["f1"] == 1.0
    assert metrics["macro_f1"] == 1 / 3


def test_v2_1_model_answer_is_guarded_before_scoring():
    from agent_pipeline_v2_1.oracle_benchmark import finalize_answer

    value = {
        "verdict": "contradiction",
        "citations": [{"evidence_id": "L1", "quote": "亚巴顿寄宿在雷的左侧心脏里。"}],
        "reason": "没有写右侧，所以矛盾。",
        "assessment": {
            "same_subject": True,
            "evidence_applicable": True,
            "relation": "direct_conflict",
            "can_both_be_true": True,
            "assumptions": [],
        },
    }

    final = finalize_answer(value, fixture_item()["lore"], "v2.1")
    assert final["verdict"] == "uncertain"
    assert final["model_verdict"] == "contradiction"
    assert final["guard_reason"] == "coexistence_not_ruled_out"


def test_comparison_report_keeps_both_prompts_and_computes_deltas():
    from agent_pipeline_v2_1.oracle_benchmark import build_comparison_report, render_comparison_markdown

    v1 = {"prompt_version": "v1", "status": "ok", "metrics": {"accuracy": 0.6, "macro_f1": 0.55, "contradiction_false_positive": 10, "contradiction_false_negative": 6}}
    v21 = {"prompt_version": "v2.1", "status": "ok", "metrics": {"accuracy": 0.75, "macro_f1": 0.7, "contradiction_false_positive": 3, "contradiction_false_negative": 8}}

    report = build_comparison_report([v1, v21], fixture_sha256="abc")
    markdown = render_comparison_markdown(report)

    assert report["comparison"]["accuracy_delta"] == 0.15
    assert report["comparison"]["contradiction_false_positive_delta"] == -7
    assert report["comparison"]["contradiction_false_negative_delta"] == 2
    assert "Judge v1" in markdown
    assert "Judge v2.1" in markdown
    assert "Oracle Retrieval" in markdown


def test_oracle_cli_defaults_lock_the_formal_protocol():
    from agent_pipeline_v2_1.cli import build_parser

    args = build_parser().parse_args(["run-oracle-judge", "--device", "cuda"])
    assert args.batch_size == 8
    assert args.repeats == 3
    assert args.prompt_versions == ["v1", "v2.1"]
    assert args.reuse_v1_from is None
