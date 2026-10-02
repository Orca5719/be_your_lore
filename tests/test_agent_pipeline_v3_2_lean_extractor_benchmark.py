import pytest

from agent_pipeline_v3_2.lean_extractor_benchmark import apply_review_decisions, reuse_exact_event_reviews, score_extraction_case_aware


def _event(event_id="E1", text="雷拥有两颗心脏"):
    return {"id": event_id, "actors": ["雷"], "event": text, "mental_state": None, "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism"}


def _row(event, system="candidate"):
    return {"case_id": "SL-001", "system": system, "result": {"benchmark_stages": {"extraction": {"events": [event]}}, "judge": {"events": [event]}}}


def test_exact_event_reuses_label_and_gold_mapping():
    dataset = {"cases": [{"id": "SL-001", "gold_facts": [{"id": "F1"}]}]}
    previous = {
        "system_event_labels": {"candidate:SL-001": {"E1": {"label": "valid_checkable", "gold_ids": ["F1"]}}},
        "event_mapping": {"candidate:SL-001": {"F1": ["E1"]}},
    }
    ledger = reuse_exact_event_reviews(dataset, [_row(_event())], [_row(_event("E7"), "lean-extractor")], previous)
    assert ledger["status"] == "reviewed"
    assert ledger["reused_event_count"] == 1
    assert ledger["event_mapping"]["SL-001"]["F1"] == ["E7"]


def test_changed_event_stays_pending():
    dataset = {"cases": [{"id": "SL-001", "gold_facts": [{"id": "F1"}]}]}
    previous = {
        "system_event_labels": {"candidate:SL-001": {"E1": {"label": "valid_checkable", "gold_ids": ["F1"]}}},
        "event_mapping": {"candidate:SL-001": {"F1": ["E1"]}},
    }
    ledger = reuse_exact_event_reviews(dataset, [_row(_event())], [_row(_event("E1", "雷有三颗心脏"), "lean-extractor")], previous)
    assert ledger["status"] == "pending_review"
    assert ledger["pending_event_count"] == 1
    assert ledger["reused_event_count"] == 0


def test_extraction_recall_counts_same_gold_id_in_different_stories_separately():
    cases = [
        {"id": "SL-001", "group": "one_conflict", "gold_facts": [{"id": "G1", "dimension": "time"}]},
        {"id": "SL-002", "group": "one_conflict", "gold_facts": [{"id": "G1", "dimension": "time"}]},
    ]
    first = _row(_event())
    second = _row(_event())
    second["case_id"] = "SL-002"
    review = {
        "system_event_labels": {
            "SL-001": {"E1": {"label": "valid_checkable"}},
            "SL-002": {"E1": {"label": "overselected"}},
        },
        "event_mapping": {"SL-001": {"G1": ["E1"]}, "SL-002": {"G1": []}},
    }
    score = score_extraction_case_aware(cases, [first, second], review)
    assert score["recall"] == {"hits": 1, "total": 2, "value": 0.5}
    assert score["by_group"]["one_conflict"]["recall"]["total"] == 2
    assert score["by_dimension"]["time"]["recall"]["total"] == 2


def test_review_decisions_must_cover_pending_events_and_update_gold_mapping():
    dataset = {"cases": [{"id": "SL-001", "gold_facts": [{"id": "G1"}]}]}
    review = {
        "pending": [{"case_id": "SL-001", "event_id": "E2", "event": _event("E2")}],
        "pending_event_count": 1, "reused_event_count": 1, "status": "pending_review",
        "system_event_labels": {"SL-001": {"E1": {"label": "valid_checkable"}, "E2": {"label": "pending"}}},
        "event_mapping": {"SL-001": {"G1": []}},
    }
    with pytest.raises(ValueError, match="exactly"):
        apply_review_decisions(dataset, review, {})
    decisions = {"SL-001": {"E2": {"label": "valid_checkable", "gold_ids": ["G1"], "note": "Reviewed from source"}}}
    resolved = apply_review_decisions(dataset, review, decisions)
    assert resolved["pending_event_count"] == 0
    assert resolved["status"] == "reviewed"
    assert resolved["event_mapping"]["SL-001"]["G1"] == ["E2"]
    with pytest.raises(ValueError, match="gold"):
        apply_review_decisions(dataset, review, {"SL-001": {"E2": {"label": "valid_checkable", "gold_ids": ["G9"], "note": "wrong"}}})
