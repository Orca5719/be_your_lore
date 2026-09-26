import pytest

from agent_pipeline_v3.review import build_review_template, reuse_exact_reviews, validate_review


DATASET = {"cases": [{"id": "C1", "gold_facts": [{"id": "G1"}], "gold_findings": [{"id": "GF1"}]}]}


def row(event="雷触碰墙壁"):
    return {"case_id": "C1", "system": "baseline", "result": {"judge": {"events": [{"id": "E1", "actors": ["雷"], "event": event, "source_ids": ["S1"], "context_ids": []}]}, "findings": [{"id": "R1", "event_ids": ["E1"], "event": event, "verdict": "uncertain", "reason": "无设定", "citations": []}]}}


def complete(review):
    key = "baseline:C1"
    review["status"] = "reviewed"; review["provenance"] = "assistant-reviewed"
    review["system_event_labels"][key]["E1"].update(label="valid_checkable", gold_ids=["G1"])
    review["system_finding_labels"][key]["R1"].update(label="true_positive", gold_finding_ids=["GF1"])
    review["reasoning_support_labels"][key]["R1"] = {"label": "supported", "note": ""}
    return review


def test_exact_reuse_and_changed_event_rejected():
    old = complete(build_review_template(DATASET, [row()]))
    reused = reuse_exact_reviews(build_review_template(DATASET, [row()]), old)
    assert reused["system_event_labels"]["baseline:C1"]["E1"]["label"] == "valid_checkable"
    changed = reuse_exact_reviews(build_review_template(DATASET, [row("雷穿过墙壁")]), old)
    assert changed["system_event_labels"]["baseline:C1"]["E1"]["label"] == "pending"


def test_complete_review_requires_every_reasoning_label():
    review = complete(build_review_template(DATASET, [row()]))
    assert validate_review(DATASET, [row()], review)["provenance"] == "assistant-reviewed"
    review["reasoning_support_labels"]["baseline:C1"]["R1"]["label"] = "pending"
    with pytest.raises(ValueError, match="reasoning"):
        validate_review(DATASET, [row()], review)
