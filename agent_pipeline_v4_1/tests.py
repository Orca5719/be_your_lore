import unittest
from unittest.mock import patch

from agent_pipeline_v4_1.audit import event_signature, score_extraction, score_end_to_end


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "S", "gold_facts": [
            {"id": "G1", "expected_verdict": "矛盾"},
            {"id": "G2", "expected_verdict": "一致"},
        ]}]
        self.extraction = {"events": [
            {"id": "E1"}, {"id": "E2"}, {"id": "E3"}, {"id": "E4"},
        ]}
        self.labels = {"S": {
            "E1": {"label": "valid_checkable", "gold_ids": ["G1"]},
            "E2": {"label": "hallucinated", "gold_ids": []},
            "E3": {"label": "overselected", "gold_ids": []},
            "E4": {"label": "duplicate", "gold_ids": []},
        }}
        for event in self.extraction["events"]:
            self.labels["S"][event["id"]]["signature"] = event_signature(event)

    def test_extraction_counts_are_disjoint(self):
        rows = [{"case_id": "S", "result": {"benchmark_stages": {"extraction": self.extraction}}}]
        scored = score_extraction(self.cases, rows, self.labels)
        self.assertEqual((scored["gold_hits"], scored["gold_total"]), (1, 2))
        self.assertEqual((scored["valid_events"], scored["hallucinated_events"],
                          scored["overselected_events"], scored["duplicate_events"]), (1, 1, 1, 1))

    def test_e2e_counts_unmapped_contradiction_as_false_positive(self):
        row = {"case_id": "S", "result": {"judge": {"items": [
            {"event_id": "E1", "status": "ok", "verdict": "contradiction"},
            {"event_id": "E2", "status": "ok", "verdict": "contradiction"},
        ]}}}
        scored = score_end_to_end(self.cases, [row], self.labels)
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (1, 1, 0))
        self.assertAlmostEqual(scored["f1"], 2 / 3)

    def test_e2e_does_not_credit_unsupported_gold_mapping(self):
        self.labels["S"]["E2"]["gold_ids"] = ["G1"]
        row = {"case_id": "S", "result": {"judge": {"items": [
            {"event_id": "E2", "status": "ok", "verdict": "contradiction"},
        ]}}}
        scored = score_end_to_end(self.cases, [row], self.labels)
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (0, 1, 1))

    def test_missing_gold_contradiction_is_false_negative(self):
        row = {"case_id": "S", "result": {"judge": {"items": [
            {"event_id": "E1", "status": "ok", "verdict": "uncertain"},
        ]}}}
        scored = score_end_to_end(self.cases, [row], self.labels)
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (0, 0, 1))

    def test_stale_event_review_is_rejected(self):
        rows = [{"case_id": "S", "result": {"benchmark_stages": {"extraction": self.extraction}}}]
        self.labels["S"]["E1"]["signature"] = "old"
        with self.assertRaisesRegex(ValueError, "stale review signature"):
            score_extraction(self.cases, rows, self.labels)

    def test_downstream_uses_frozen_stage_order_and_settings(self):
        from agent_pipeline_v4_1.cli import _run_one

        extraction = {"events": [{"id": "E1"}]}
        retrieval = {"status": "ok"}
        judge = {"status": "ok"}
        report = {"status": "ok", "judge": judge}
        with patch("agent_pipeline_v2_1.story_pipeline.retrieve_story_events", return_value=retrieval) as retrieve, \
             patch("agent_pipeline_v2_1.story_pipeline.judge_story_events", return_value=judge) as judge_call, \
             patch("agent_pipeline_v2_1.story_pipeline.build_story_report", return_value=report) as report_call:
            result = _run_one(extraction, "llm", "retriever")
        retrieve.assert_called_once_with(extraction, "retriever", "hybrid", False, 5)
        judge_call.assert_called_once_with(retrieval, "llm", 8)
        report_call.assert_called_once_with(judge)
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
