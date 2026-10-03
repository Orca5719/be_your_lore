import json
import unittest

from agent_pipeline_v2.extractor import extract_events as baseline_extract
from agent_pipeline_v3_3.coverage import account_coverage
from agent_pipeline_v3_3.extractor import extract_events
from agent_pipeline_v3_3.systems import build_system
from agent_pipeline_v3_3.cli import build_parser


class FakeLLM:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self, answer):
        self.answer = answer
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens):
        self.last_generation = {"seconds": 0.01, "input_tokens": 20, "generated_tokens": 10}
        return json.dumps(self.answer, ensure_ascii=False)


class CoverageTests(unittest.TestCase):
    def test_accounting_is_programmatic_and_keeps_missing_pending(self):
        result = account_coverage(
            ["S1", "S2", "S3"],
            [{"source_ids": ["S1"]}],
            [{"source_id": "S2", "reason": "routine"}],
            [],
        )
        self.assertEqual(result["event_span_ids"], ["S1"])
        self.assertEqual(result["ignored_span_ids"], ["S2"])
        self.assertEqual(result["uncovered_span_ids"], ["S3"])
        self.assertFalse(result["complete"])

    def test_conflicting_dispositions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复处置"):
            account_coverage(["S1"], [{"source_ids": ["S1"]}], [{"source_id": "S1"}], [])

    def test_out_of_scope_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不存在"):
            account_coverage(["S1"], [{"source_ids": ["S2"]}], [], [])

    def test_accounting_lists_uncovered_ids_in_target_order(self):
        ids = [f"S{number}" for number in range(1, 13)]
        result = account_coverage(ids, [], [], [])
        self.assertEqual(result["uncovered_span_ids"], ids)

    def test_restored_extractor_matches_benchmark_3_contract(self):
        answer = {
            "events": [{"actors": ["雷"], "event": "雷拥有两颗心脏", "mental_state": None,
                        "explicit": True, "modality": "observed", "conditions": [],
                        "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism"}],
            "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
            "non_event_span_ids": ["S3"],
        }
        story = "雷拥有两颗心脏。他喝了一口水。随后，"
        old = baseline_extract(story, llm=FakeLLM(answer))
        new = extract_events(story, llm=FakeLLM(answer))
        for field in ("schema_version", "prompt_version", "events", "ignored_spans",
                      "non_event_span_ids", "uncovered_span_ids", "status"):
            self.assertEqual(new[field], old[field])
        self.assertTrue(new["coverage_accounting"]["complete"])
        self.assertEqual(new["coverage_accounting"]["target_count"], 3)

    def test_candidate_system_uses_restored_extractor(self):
        system = build_system(object(), object(), device="cpu")
        self.assertEqual(system.config.retrieval, "hybrid")
        self.assertEqual(system.config.judge, "v2.1")
        self.assertIs(system.extractor, extract_events)

    def test_independent_cli_has_validate_and_run(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["validate"]).command, "validate")
        self.assertEqual(parser.parse_args(["run", "--device", "cuda"]).command, "run")


if __name__ == "__main__":
    unittest.main()
