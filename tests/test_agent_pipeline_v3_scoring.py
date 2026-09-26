import unittest

from agent_pipeline_v3.scoring import score_system


def fixture():
    facts = [
        {"id": "G1", "expected_verdict": "矛盾", "minimum_evidence_sets": [["L1"]]},
        {"id": "G2", "expected_verdict": "矛盾", "minimum_evidence_sets": [["L2"]]},
        {"id": "G3", "expected_verdict": "矛盾", "minimum_evidence_sets": [["L3"]]},
        {"id": "G4", "expected_verdict": "不确定", "minimum_evidence_sets": []},
    ]
    case = {"id": "C1", "gold_facts": facts, "gold_findings": []}
    events = [{"id": x} for x in ("E1", "E3", "E4")]
    retrieval = [
        {"event_id": "E1", "status": "ok", "evidence": [{"id": "L1"}]},
        {"event_id": "E3", "status": "ok", "evidence": [{"id": "wrong"}]},
        {"event_id": "E4", "status": "ok", "evidence": []},
    ]
    items = [
        {"event_id": "E1", "status": "ok", "verdict": "contradiction"},
        {"event_id": "E3", "status": "ok", "verdict": "uncertain"},
        {"event_id": "E4", "status": "ok", "verdict": "contradiction"},
    ]
    row = {"case_id": "C1", "status": "ok", "result": {"judge": {"events": events, "items": items, "retrieval": {"items": retrieval}}, "findings": []}}
    review = {
        "event_mapping": {"C1": {"G1": ["E1"], "G2": [], "G3": ["E3"], "G4": ["E4"]}},
        "system_event_labels": {"C1": {eid: {"label": "valid_checkable"} for eid in ("E1", "E3", "E4")}},
        "system_finding_labels": {"C1": {}},
        "reasoning_support_labels": {"C1": {"E4": {"label": "unsupported"}}},
    }
    return [case], [row], review


class BenchmarkV3ScoringTests(unittest.TestCase):
    def test_requested_metrics_keep_counts_and_denominators(self):
        cases, rows, review = fixture()
        score = score_system(cases, rows, review, k=5)
        self.assertEqual(score["extraction"], {"hits": 3, "misses": 1, "total": 4, "recall": 0.75})
        self.assertEqual(score["retrieval"]["hits"], 1)
        self.assertEqual(score["retrieval"]["misses"], 1)
        self.assertEqual(score["retrieval"]["eligible"], 2)
        self.assertEqual(score["judge"]["fp"], 1)
        self.assertEqual(score["judge"]["fn_total"], 2)
        self.assertEqual(score["judge"]["fn_with_complete_evidence"], 0)
        self.assertEqual(score["unsupported_reasoning"], {"count": 1, "eligible": 3, "rate": 1 / 3})
        self.assertEqual(score["end_to_end_conflict"], {"tp": 1, "fp": 1, "fn": 2, "precision": 0.5, "recall": 1 / 3, "f1": 0.4})

    def test_judge_fn_with_complete_evidence_is_separate_from_retrieval_miss(self):
        cases, rows, review = fixture()
        rows[0]["result"]["judge"]["retrieval"]["items"][1]["evidence"] = [{"id": "L3"}]
        score = score_system(cases, rows, review)
        self.assertEqual(score["retrieval"]["misses"], 0)
        self.assertEqual(score["judge"]["fn_with_complete_evidence"], 1)
        self.assertEqual(score["judge"]["fn_total"], 2)

    def test_zero_denominators_are_none(self):
        score = score_system([], [], {"event_mapping": {}, "system_event_labels": {}})
        self.assertIsNone(score["extraction"]["recall"])
        self.assertIsNone(score["retrieval"]["recall_at_5"])
        self.assertIsNone(score["judge"]["accuracy"])
        self.assertIsNone(score["end_to_end_conflict"]["precision"])


if __name__ == "__main__":
    unittest.main()
