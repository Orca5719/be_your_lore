import unittest

from agent_pipeline_v2.scoring.detection import score_detection


CASES = [{
    "id": "S1", "group": "multi_conflict",
    "gold_facts": [
        {"id": "G1", "expected_verdict": "矛盾", "reportable": True},
        {"id": "G2", "expected_verdict": "矛盾", "reportable": True},
    ],
    "gold_findings": [{"id": "F1", "gold_fact_ids": ["G1", "G2"]}],
}]
RUNS = [{"case_id": "S1", "result": {
    "findings": [
        {"id": "R1", "event_id": "E1", "verdict": "contradiction"},
        {"id": "R2", "event_id": "E2", "verdict": "contradiction"},
    ],
    "judge": {"events": [{"id": "E1"}, {"id": "E2"}], "items": [
        {"event_id": "E1", "status": "ok", "verdict": "contradiction"},
        {"event_id": "E2", "status": "ok", "verdict": "contradiction"},
    ]},
}}]
REVIEW = {
    "event_mapping": {"S1": {"G1": ["E1"], "G2": ["E1"]}},
    "system_event_labels": {"S1": {"E1": {"label": "valid_checkable"}, "E2": {"label": "duplicate", "duplicate_of": "E1"}}},
    "finding_mapping": {"S1": {"F1": ["R1", "R2"]}},
    "system_finding_labels": {"S1": {
        "R1": {"label": "true_positive", "gold_finding_ids": ["F1"]},
        "R2": {"label": "duplicate", "gold_finding_ids": ["F1"], "duplicate_of": "R1"},
    }},
}


class DetectionMetricTests(unittest.TestCase):
    def test_duplicate_conflict_gets_one_tp_and_one_duplicate(self):
        score = score_detection(CASES, RUNS, REVIEW)
        self.assertEqual(score["conflict"], {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(score["duplicate_finding_rate"], {"count": 1, "total": 2, "value": 0.5})

    def test_false_positive_and_unsupported_claim_are_counted(self):
        runs = [{"case_id": "S1", "result": {"findings": [
            {"id": "R1", "event_id": "E1", "verdict": "contradiction"},
            {"id": "R3", "event_id": "E3", "verdict": "contradiction"},
        ], "judge": {"events": [{"id": "E1"}], "items": [{"event_id": "E1", "status": "ok", "verdict": "contradiction"}]}}}]
        review = {**REVIEW, "finding_mapping": {"S1": {"F1": ["R1"]}}, "system_finding_labels": {"S1": {
            "R1": {"label": "true_positive", "gold_finding_ids": ["F1"]},
            "R3": {"label": "unsupported", "gold_finding_ids": [], "note": "无依据"},
        }}}
        score = score_detection(CASES, runs, review)
        self.assertEqual(score["conflict"]["tp"], 1)
        self.assertEqual(score["conflict"]["fp"], 1)
        self.assertEqual(score["unsupported_report_claim_rate"], {"count": 1, "total": 2, "value": 0.5})
        self.assertEqual(score["story_any_false_alarm_rate"], {"count": 1, "total": 1, "value": 1.0})

    def test_report_omission_and_partial_story_become_fn_without_crashing(self):
        runs = [{"case_id": "S1", "result": {"status": "partial", "findings": [], "judge": {"events": [{"id": "E1"}], "items": [{"event_id": "E1", "status": "ok", "verdict": "contradiction"}]}}}]
        review = {"event_mapping": {"S1": {"G1": ["E1"], "G2": ["E1"]}}, "system_event_labels": {"S1": {"E1": {"label": "valid_checkable"}}}, "finding_mapping": {"S1": {"F1": []}}, "system_finding_labels": {"S1": {}}}
        score = score_detection(CASES, runs, review)
        self.assertEqual(score["conflict"]["fn"], 1)
        self.assertEqual(score["report_omission_rate"], {"count": 2, "total": 2, "value": 1.0})
        self.assertEqual(score["story_exact_match"], {"count": 0, "total": 1, "value": 0.0})


if __name__ == "__main__":
    unittest.main()
