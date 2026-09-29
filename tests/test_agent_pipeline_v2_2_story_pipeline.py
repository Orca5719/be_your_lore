import unittest

from agent_pipeline_v2_2.schema import SystemConfig, validate_run_row
from agent_pipeline_v2_2.story_pipeline import PipelineSystem, process_story


class BenchmarkV3StoryPipelineTests(unittest.TestCase):
    def test_normalized_row_has_all_stage_metrics_and_provenance(self):
        system = PipelineSystem(
            SystemConfig.candidate(),
            lambda text, progress=None: (
                {"status": "ok", "events": []},
                {"status": "ok", "method": "hybrid", "metadata_filter": False},
                {"status": "ok", "items": []},
                {"status": "ok", "findings": []},
            ),
        )
        row = process_story(system, "C1", "story")
        validate_run_row(row)
        self.assertEqual(row["system"], "candidate")
        self.assertEqual(row["result"]["retrieval_provenance"], {"method": "hybrid", "metadata_filter": False, "top_k": 5})
        self.assertEqual(set(row["stage_metrics"]), {"extraction", "retrieval", "judge", "report", "total"})
        self.assertTrue(all(value["seconds"] >= 0 for value in row["stage_metrics"].values()))

    def test_stage_exception_becomes_error_row_without_hiding_cause(self):
        def fail(text, progress=None):
            raise ValueError("broken stage")
        row = process_story(PipelineSystem(SystemConfig.baseline(), fail), "C2", "story")
        self.assertEqual(row["status"], "error")
        self.assertIn("broken stage", row["error"])
        validate_run_row(row)


if __name__ == "__main__":
    unittest.main()
