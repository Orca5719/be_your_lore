import copy
import unittest

from agent_pipeline_v2.review import build_review_template, propose_review, validate_review


DATASET = {"cases": [{"id": "S1", "gold_facts": [{"id": "G1", "source_anchors": [{"text": "雷拥有两颗心脏。", "start": 0, "end": 8}], "normalized_fact": "雷拥有两颗心脏", "subject": "雷", "expected_verdict": "一致"}], "gold_findings": []}]}
ROWS = [{"case_id": "S1", "result": {"judge": {"events": [{"id": "E1", "actors": ["雷"], "event": "雷拥有两颗心脏", "source_ids": ["S1"]}]}, "findings": [{"id": "R1", "event_ids": ["E1"], "verdict": "consistent"}]}}]


class ReviewLedgerTests(unittest.TestCase):
    def test_template_labels_every_event_and_finding(self):
        review = build_review_template(DATASET, ROWS)
        self.assertEqual(review["status"], "pending_human_review")
        self.assertIn("E1", review["system_event_labels"]["S1"])
        self.assertEqual(review["system_event_labels"]["S1"]["E1"]["label"], "pending")
        self.assertIn("R1", review["system_finding_labels"]["S1"])
        self.assertEqual(review["event_mapping"]["S1"]["G1"], [])

    def test_proposals_use_anchor_and_subject_overlap_without_reviewing(self):
        proposals = propose_review(DATASET, ROWS)
        self.assertEqual(proposals["event_mapping"]["S1"]["G1"][0]["event_id"], "E1")
        self.assertEqual(proposals["status"], "pending_human_review")

    def test_validation_rejects_unreviewed_event(self):
        review = build_review_template(DATASET, ROWS)
        review["status"] = "reviewed"
        with self.assertRaisesRegex(ValueError, "未审核事件"):
            validate_review(DATASET, ROWS, review, require_complete=True)
        self.assertEqual(validate_review(DATASET, ROWS, review, require_complete=False)["cases"], 1)

    def test_validation_accepts_complete_review_and_requires_duplicate_origin(self):
        review = build_review_template(DATASET, ROWS)
        review["event_mapping"]["S1"]["G1"] = ["E1"]
        review["system_event_labels"]["S1"]["E1"] = {"label": "valid_checkable", "gold_ids": ["G1"], "note": "直接对应"}
        review["system_finding_labels"]["S1"]["R1"] = {"label": "false_positive", "gold_finding_ids": [], "note": "一致事实"}
        review["status"] = "reviewed"
        self.assertEqual(validate_review(DATASET, ROWS, review, require_complete=True)["status"], "reviewed")

        duplicate = copy.deepcopy(review)
        duplicate["system_event_labels"]["S1"]["E1"] = {"label": "duplicate", "gold_ids": ["G1"], "note": "重复", "duplicate_of": "missing"}
        with self.assertRaisesRegex(ValueError, "duplicate_of"):
            validate_review(DATASET, ROWS, duplicate, require_complete=True)


if __name__ == "__main__":
    unittest.main()
