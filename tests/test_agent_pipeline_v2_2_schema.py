import unittest

from agent_pipeline_v2_2.schema import (
    RunIdentity,
    SystemConfig,
    stable_digest,
    validate_run_row,
    validate_system_config,
)


class BenchmarkV3SchemaTests(unittest.TestCase):
    def test_fixed_baseline_and_candidate_are_valid(self):
        baseline = SystemConfig.baseline()
        candidate = SystemConfig.candidate()
        self.assertEqual((baseline.retrieval, baseline.judge), ("dense", "v1"))
        self.assertEqual((candidate.retrieval, candidate.judge), ("hybrid", "v2.1"))
        self.assertFalse(candidate.metadata_filter)
        validate_system_config(baseline)
        validate_system_config(candidate)

    def test_candidate_rejects_metadata_filter_or_wrong_fixed_values(self):
        with self.assertRaisesRegex(ValueError, "metadata filter"):
            validate_system_config(SystemConfig("candidate", "hybrid", "v2.1", True, 5, 8))
        with self.assertRaisesRegex(ValueError, "Top-K"):
            validate_system_config(SystemConfig("candidate", "hybrid", "v2.1", False, 3, 8))
        with self.assertRaisesRegex(ValueError, "batch"):
            validate_system_config(SystemConfig("baseline", "dense", "v1", False, 5, 4))

    def test_digest_is_order_independent_for_mapping_keys(self):
        self.assertEqual(stable_digest({"a": 1, "b": [2]}), stable_digest({"b": [2], "a": 1}))
        self.assertNotEqual(stable_digest({"a": 1}), stable_digest({"a": 2}))

    def test_run_identity_rejects_duplicate_case_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            RunIdentity.create("dataset", SystemConfig.baseline(), ["SL-1", "SL-1"])

    def test_run_row_requires_stage_metrics_and_valid_status(self):
        valid = {
            "case_id": "SL-1",
            "system": "baseline",
            "status": "ok",
            "result": {},
            "stage_metrics": {
                name: {"seconds": 0.1} for name in ("extraction", "retrieval", "judge", "report", "total")
            },
        }
        validate_run_row(valid)
        with self.assertRaisesRegex(ValueError, "status"):
            validate_run_row(dict(valid, status="uncertain"))
        with self.assertRaisesRegex(ValueError, "stage metrics"):
            validate_run_row(dict(valid, stage_metrics={"total": {"seconds": 1.0}}))


if __name__ == "__main__":
    unittest.main()
