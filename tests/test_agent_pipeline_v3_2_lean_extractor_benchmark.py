from agent_pipeline_v3_2.lean_extractor_benchmark import reuse_exact_event_reviews


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
