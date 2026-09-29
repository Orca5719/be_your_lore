import unittest

from agent_pipeline_v2.scoring.retrieval import score_retrieval


CASES = [{
    "id": "S1",
    "group": "one_conflict",
    "gold_facts": [
        {"id": "G1", "dimension": "physical_rule", "relevant_lore_ids": ["C2", "C3", "C4"], "minimum_evidence_sets": [["C2", "C3"], ["C4"]]},
        {"id": "G2", "dimension": "identity", "relevant_lore_ids": ["C5"], "minimum_evidence_sets": []},
    ],
}]
ROWS = [{"case_id": "S1", "result": {"judge": {"events": [{"id": "E1"}, {"id": "E2"}], "retrieval": {"items": [
    {"event_id": "E1", "status": "ok", "evidence": [{"id": "NOISE"}, {"id": "C2"}, {"id": "C3"}]},
    {"event_id": "E2", "status": "ok", "evidence": [{"id": "C5"}, {"id": "NOISE2"}]},
]}}}}]
REVIEW = {
    "event_mapping": {"S1": {"G1": ["E1"], "G2": ["E2"]}},
    "system_event_labels": {"S1": {
        "E1": {"label": "valid_checkable", "gold_ids": ["G1"]},
        "E2": {"label": "valid_checkable", "gold_ids": ["G2"]},
    }},
    "retrieval_relevance_overrides": {},
}


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_uses_minimum_sets_and_mrr_first_relevant_rank(self):
        score = score_retrieval(CASES, ROWS, REVIEW, k=5)
        self.assertEqual(score["recall_at_k"], {"hits": 1, "total": 1, "value": 1.0})
        self.assertEqual(score["mrr"], {"sum_reciprocal_rank": 0.5, "total": 1, "value": 0.5})

    def test_precision_and_noise_count_every_returned_chunk_once(self):
        score = score_retrieval(CASES, ROWS, REVIEW, k=5)
        self.assertEqual(score["precision_at_k"], {"relevant": 3, "total": 5, "value": 0.6})
        self.assertEqual(score["noise_rate_at_k"], {"irrelevant": 2, "total": 5, "value": 0.4})
        self.assertEqual(score["event_diagnostics"]["scored_events"], 2)

    def test_missing_retrieval_row_is_exposed_without_becoming_false_evidence(self):
        rows = [{"case_id": "S1", "result": {"judge": {"events": [{"id": "E1"}], "retrieval": {"items": []}}}}]
        review = {"event_mapping": {"S1": {"G1": ["E1"], "G2": []}}, "system_event_labels": {"S1": {"E1": {"label": "valid_checkable", "gold_ids": ["G1"]}}}, "retrieval_relevance_overrides": {}}
        score = score_retrieval(CASES, rows, review, k=5)
        self.assertEqual(score["recall_at_k"], {"hits": 0, "total": 1, "value": 0.0})
        self.assertEqual(score["event_diagnostics"]["missing_retrieval_events"], 1)
        self.assertIsNone(score["precision_at_k"]["value"])

    def test_dimension_breakdown_and_zero_denominator(self):
        score = score_retrieval(CASES, ROWS, REVIEW, k=5)
        self.assertIn("physical_rule", score["by_dimension"])
        self.assertEqual(score["by_dimension"]["identity"]["recall_at_k"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
