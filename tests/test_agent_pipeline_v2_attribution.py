import unittest

from agent_pipeline_v2.attribution import build_error_attribution


class AttributionTests(unittest.TestCase):
    def test_extractor_hallucination_remains_primary_when_judge_accepts_it(self):
        dataset = [{"id": "S1", "gold_facts": [{"id": "G1", "expected_verdict": "矛盾", "minimum_evidence_sets": [["C1"]]}], "gold_findings": []}]
        rows = [{"case_id": "S1", "result": {"judge": {"events": [{"id": "E1"}], "items": [{"event_id": "E1", "status": "ok", "verdict": "consistent"}], "retrieval": {"items": []}}, "findings": []}}]
        review = {"event_mapping": {"S1": {"G1": []}}, "system_event_labels": {"S1": {"E1": {"label": "hallucinated", "gold_ids": [], "note": "原文不支持"}}}, "system_finding_labels": {"S1": {}}, "error_attribution": {}}
        errors = build_error_attribution({"cases": dataset}, rows, review, {})["errors"]
        self.assertEqual(errors[0]["error_type"], "EXTRACTION_HALLUCINATION")
        self.assertEqual(errors[0]["primary_stage"], "extractor")
        self.assertIn("judge", errors[0]["contributing_stages"])

    def test_extraction_miss_precedes_downstream_judge_absence(self):
        dataset = [{"id": "S1", "gold_facts": [{"id": "G1", "expected_verdict": "矛盾", "minimum_evidence_sets": [["C1"]]}], "gold_findings": []}]
        rows = [{"case_id": "S1", "result": {"judge": {"events": [], "items": [], "retrieval": {"items": []}}, "findings": []}}]
        review = {"event_mapping": {"S1": {"G1": []}}, "system_event_labels": {"S1": {}}, "system_finding_labels": {"S1": {}}, "error_attribution": {}}
        errors = build_error_attribution({"cases": dataset}, rows, review, {})["errors"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "EXTRACTION_MISS")
        self.assertEqual(errors[0]["primary_stage"], "extractor")

    def test_reviewed_attribution_overrides_automatic_suggestion(self):
        dataset = [{"id": "S1", "gold_facts": [], "gold_findings": []}]
        rows = [{"case_id": "S1", "result": {"judge": {"events": [{"id": "E1"}], "items": [], "retrieval": {"items": []}}, "findings": []}}]
        review = {"event_mapping": {"S1": {}}, "system_event_labels": {"S1": {"E1": {"label": "overselected", "gold_ids": [], "note": "普通动作"}}}, "system_finding_labels": {"S1": {}}, "error_attribution": {"S1": {"A1": {"review_status": "reviewed", "error_type": "EXTRACTION_OVERSELECT", "primary_stage": "extractor", "reviewer_note": "确认"}}}}
        result = build_error_attribution({"cases": dataset}, rows, review, {})
        self.assertEqual(result["errors"][0]["review_status"], "reviewed")
        self.assertEqual(result["errors"][0]["reviewer_note"], "确认")


if __name__ == "__main__":
    unittest.main()
