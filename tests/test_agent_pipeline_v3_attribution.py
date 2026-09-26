import unittest

from agent_pipeline_v3.attribution import attribute_first_failures
from tests.test_agent_pipeline_v3_scoring import fixture


class BenchmarkV3AttributionTests(unittest.TestCase):
    def test_first_failure_is_mutually_exclusive_and_ordered(self):
        cases, rows, review = fixture()
        result = attribute_first_failures(cases, rows, review, k=5)
        by_gold = {row["gold_fact_id"]: row["primary_cause"] for row in result["items"]}
        self.assertEqual(by_gold, {"G2": "extraction_miss", "G3": "retrieval_miss"})
        self.assertEqual(result["counts"], {"extraction_miss": 1, "retrieval_miss": 1})

    def test_complete_evidence_moves_failure_to_judge(self):
        cases, rows, review = fixture()
        rows[0]["result"]["judge"]["retrieval"]["items"][1]["evidence"] = [{"id": "L3"}]
        result = attribute_first_failures(cases, rows, review)
        by_gold = {row["gold_fact_id"]: row["primary_cause"] for row in result["items"]}
        self.assertEqual(by_gold["G3"], "judge_fn")

    def test_execution_failure_wins_before_extraction(self):
        cases, rows, review = fixture()
        rows[0]["status"] = "error"
        rows[0]["result"] = {}
        result = attribute_first_failures(cases, rows, review)
        self.assertTrue(result["items"])
        self.assertEqual({x["primary_cause"] for x in result["items"]}, {"execution_failure"})

    def test_correct_judge_without_report_finding_is_report_miss(self):
        cases, rows, review = fixture()
        cases[0]["gold_findings"] = [{"id": "F1", "gold_fact_ids": ["G1"]}]
        review["finding_mapping"] = {"C1": {"F1": []}}
        result = attribute_first_failures(cases, rows, review)
        by_gold = {row["gold_fact_id"]: row["primary_cause"] for row in result["items"]}
        self.assertEqual(by_gold["G1"], "report_miss")


if __name__ == "__main__":
    unittest.main()
