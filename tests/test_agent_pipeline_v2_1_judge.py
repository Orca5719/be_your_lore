from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return [{"id": "lore-1", "text": "雷的左侧心脏寄宿亚巴顿。", "heading_path": ["雷", "生理结构"]}]


def answer(verdict="contradiction", **assessment_changes):
    assessment = {
        "same_subject": True,
        "evidence_applicable": True,
        "relation": "direct_conflict",
        "can_both_be_true": False,
        "assumptions": [],
    }
    assessment.update(assessment_changes)
    return {
        "verdict": verdict,
        "citations": [{"evidence_id": "L1", "quote": "雷的左侧心脏寄宿亚巴顿。"}],
        "reason": "同一时间、同一心脏位置不能由两个不同天使唯一寄宿。",
        "assessment": assessment,
    }


def test_validator_accepts_explicitly_incompatible_contradiction():
    from agent_pipeline_v2_1.judge import validate_answer

    validate_answer(answer(), evidence())


@pytest.mark.parametrize(
    "changes",
    [
        {"can_both_be_true": True},
        {"can_both_be_true": None},
        {"same_subject": False},
        {"evidence_applicable": False},
        {"relation": "insufficient"},
        {"assumptions": ["假定寄宿关系具有唯一性"]},
    ],
)
def test_validator_rejects_contradiction_without_impossibility_proof(changes):
    from agent_pipeline_v2_1.judge import validate_answer

    with pytest.raises(ValueError, match="contradiction"):
        validate_answer(answer(**changes), evidence())


def test_scope_guard_downgrades_model_contradiction_when_claims_can_coexist():
    from agent_pipeline_v2_1.judge import apply_scope_guard

    guarded = apply_scope_guard(answer(can_both_be_true=True))

    assert guarded["verdict"] == "uncertain"
    assert guarded["model_verdict"] == "contradiction"
    assert guarded["guard_reason"] == "coexistence_not_ruled_out"


def test_new_information_remains_uncertain_without_citations():
    from agent_pipeline_v2_1.judge import apply_scope_guard, validate_answer

    value = {
        "verdict": "uncertain",
        "citations": [],
        "reason": "候选设定没有说明该能力，缺失不等于禁止。",
        "assessment": {
            "same_subject": True,
            "evidence_applicable": False,
            "relation": "insufficient",
            "can_both_be_true": True,
            "assumptions": [],
        },
    }

    validate_answer(value, evidence())
    assert apply_scope_guard(value)["verdict"] == "uncertain"


def test_prompt_centers_the_cannot_both_be_true_rule():
    prompt = (ROOT / "agent_pipeline_v2_1" / "prompts" / "judge_v2_1.txt").read_text(encoding="utf-8")

    assert "无法同时为真" in prompt
    assert "没有提到" in prompt
    assert "can_both_be_true" in prompt
