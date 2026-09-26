import unittest

from agent_pipeline_v2.scoring.judge import classification_metrics, score_pipeline_judge


class JudgeMetricTests(unittest.TestCase):
    def test_classification_reports_confusion_and_macro_f1(self):
        rows = [
            {"expected": "一致", "predicted": "一致"},
            {"expected": "矛盾", "predicted": "不确定"},
            {"expected": "不确定", "predicted": "不确定"},
        ]
        result = classification_metrics(rows)
        self.assertEqual(result["accuracy"], {"correct": 2, "total": 3, "value": 2 / 3})
        self.assertEqual(result["confusion_matrix"]["矛盾"]["不确定"], 1)
        self.assertAlmostEqual(result["macro_f1"], (1 + 0 + 2 / 3) / 3)
        self.assertEqual(set(result["per_class"]), {"一致", "矛盾", "不确定"})

    def test_pipeline_judge_keeps_extraction_and_strict_evidence_denominators_separate(self):
        cases = [{"id": "S1", "gold_facts": [
            {"id": "G1", "expected_verdict": "一致", "minimum_evidence_sets": [["C1"]]},
            {"id": "G2", "expected_verdict": "矛盾", "minimum_evidence_sets": [["C2"]]},
            {"id": "G3", "expected_verdict": "不确定", "minimum_evidence_sets": []},
        ]}]
        rows = [{"case_id": "S1", "result": {"judge": {
            "events": [{"id": "E1"}, {"id": "E2"}],
            "items": [
                {"event_id": "E1", "status": "ok", "verdict": "consistent"},
                {"event_id": "E2", "status": "ok", "verdict": "contradiction"},
            ],
            "retrieval": {"items": [
                {"event_id": "E1", "status": "ok", "evidence": [{"id": "C1"}]},
                {"event_id": "E2", "status": "ok", "evidence": [{"id": "NOISE"}]},
            ]},
        }}}]
        review = {"event_mapping": {"S1": {"G1": ["E1"], "G2": ["E2"], "G3": []}}, "system_event_labels": {"S1": {"E1": {"label": "valid_checkable"}, "E2": {"label": "valid_checkable"}}}, "reasoning_support_labels": {"S1": {"E2": {"label": "unsupported"}}}}
        result = score_pipeline_judge(cases, rows, review)
        self.assertEqual(result["pipeline"]["accuracy"]["total"], 2)
        self.assertEqual(result["strict_evidence"]["accuracy"]["total"], 1)
        self.assertEqual(result["extraction_misses"], 1)
        self.assertEqual(result["unsupported_reasoning"], {"count": 1, "total": 2, "value": 0.5})

    def test_undefined_classification_ratios_are_null(self):
        result = classification_metrics([])
        self.assertEqual(result["accuracy"], {"correct": 0, "total": 0, "value": None})
        self.assertIsNone(result["macro_f1"])


if __name__ == "__main__":
    unittest.main()
