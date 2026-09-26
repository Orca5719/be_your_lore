import unittest

from agent_pipeline_v2.summary import build_benchmark_summary, render_benchmark_markdown


class SummaryTests(unittest.TestCase):
    def test_summary_contains_headline_stage_tables_and_performance(self):
        story_quality = {
            "extraction": {"recall": {"value": 0.5}, "precision": {"value": 0.25}, "hallucination_rate": {"value": None}},
            "retrieval": {"recall_at_k": {"value": 0.75}},
            "judge": {"pipeline": {"accuracy": {"value": 1.0}, "macro_f1": 0.8}},
            "detection": {"conflict": {"f1": 0.4}, "story_exact_match": {"value": 0.2}},
        }
        attribution = {"counts": {"EXTRACTION_HALLUCINATION": 2}}
        judge_report = {"status": "ok", "results": [{"batch_size": 8, "repeat_count": 3, "distributions": {"total_judge_seconds": {"median": 10.0, "min": 8.0, "max": 12.0, "stddev": 1.63}}}]}
        run_status = {"status": "ok", "tracks": {"stories": "ok", "judge": "ok"}}
        summary = build_benchmark_summary(story_quality, attribution, judge_report, run_status)
        self.assertEqual(summary["headline"]["conflict_f1"], 0.4)
        self.assertEqual(summary["errors"]["counts"]["EXTRACTION_HALLUCINATION"], 2)
        self.assertEqual(summary["performance"]["batch_size"], 8)
        markdown = render_benchmark_markdown(summary)
        self.assertIn("Conflict F1", markdown)
        self.assertIn("N/A", markdown)

    def test_none_is_rendered_as_na(self):
        summary = build_benchmark_summary({"detection": {"conflict": {"f1": None}}}, {}, {}, {})
        markdown = render_benchmark_markdown(summary)
        self.assertIn("N/A", markdown)
        self.assertNotIn("None", markdown)


if __name__ == "__main__":
    unittest.main()
