import unittest

from agent_pipeline_v2.scoring.extraction import score_extraction


CASES = [
    {"id": "S1", "group": "one_conflict", "dimensions": ["physical_rule"], "gold_facts": [{"id": "G1", "expected_verdict": "矛盾", "dimension": "physical_rule"}, {"id": "G2", "expected_verdict": "一致", "dimension": "physical_rule"}]},
    {"id": "S2", "group": "zero_conflict", "dimensions": [], "gold_facts": []},
]
ROWS = [
    {"case_id": "S1", "result": {"judge": {"events": [{"id": "E1"}, {"id": "E2"}, {"id": "E3"}]} }},
    {"case_id": "S2", "result": {"judge": {"events": []}}},
]
REVIEW = {
    "status": "reviewed",
    "event_mapping": {"S1": {"G1": ["E1", "E3"], "G2": ["E2"]}, "S2": {}},
    "system_event_labels": {"S1": {
        "E1": {"label": "valid_checkable", "gold_ids": ["G1"], "note": ""},
        "E2": {"label": "overselected", "gold_ids": [], "note": "普通动作"},
        "E3": {"label": "duplicate", "gold_ids": ["G1"], "duplicate_of": "E1", "note": "重复"},
    }, "S2": {}},
}


class ExtractionMetricTests(unittest.TestCase):
    def test_extra_events_reduce_precision_and_split_error_rates(self):
        result = score_extraction(CASES, ROWS, REVIEW)
        self.assertEqual(result["recall"], {"hits": 1, "total": 2, "value": 0.5})
        self.assertEqual(result["precision"], {"hits": 1, "total": 3, "value": 1 / 3})
        self.assertEqual(result["hallucination_rate"], {"count": 0, "total": 3, "value": 0.0})
        self.assertEqual(result["overselection_rate"], {"count": 1, "total": 3, "value": 1 / 3})
        self.assertEqual(result["duplicate_rate"], {"count": 1, "total": 3, "value": 1 / 3})

    def test_zero_gold_and_zero_events_have_null_ratios(self):
        result = score_extraction(CASES[1:], ROWS[1:], {"event_mapping": {"S2": {}}, "system_event_labels": {"S2": {}}})
        self.assertEqual(result["recall"], {"hits": 0, "total": 0, "value": None})
        self.assertEqual(result["precision"], {"hits": 0, "total": 0, "value": None})

    def test_per_group_and_dimension_breakdowns_are_exposed(self):
        result = score_extraction(CASES, ROWS, REVIEW)
        self.assertIn("one_conflict", result["by_group"])
        self.assertIn("physical_rule", result["by_dimension"])
        self.assertEqual(result["by_group"]["zero_conflict"]["precision"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
