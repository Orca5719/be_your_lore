import json
import contextlib
import io
import unittest

from agent_pipeline_v2.benchmark import aggregate_repeats, benchmark_markdown, judge_fixture_items, score_judge_results, score_story_quality
from agent_pipeline_v2.benchmark_worker import attach_runtime_metrics
from agent_pipeline_v2.benchmark_cli import main as benchmark_main
from agent_pipeline_v2 import benchmark_cli


def items():
    result = []
    for index, (label, code) in enumerate((("一致", "consistent"), ("矛盾", "contradiction"), ("不确定", "uncertain")), 1):
        result.append({
            "fixture_id": f"F{index}",
            "fact": f"事实{index}",
            "actors": ["雷"],
            "story_context": f"事实{index}。",
            "expected_verdict": label,
            "expected_verdict_code": code,
            "lore": [{"id": f"C{index}", "text": f"设定{index}", "heading_path": ["测试"]}],
        })
    return result


def answer(verdict):
    relation = {"consistent": "direct_support", "contradiction": "direct_conflict", "uncertain": "insufficient"}[verdict]
    return json.dumps({
        "verdict": verdict,
        "citations": [] if verdict == "uncertain" else [{"evidence_id": "L1", "quote": "设定"}],
        "reason": "测试判断",
        "assessment": {"same_subject": True, "evidence_applicable": True, "relation": relation, "assumptions": []},
    }, ensure_ascii=False)


class LLM:
    device = "cpu"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.batch_sizes = []
        self.last_batch_generation = {}

    def _generate_batch(self, messages, max_new_tokens):
        self.batch_sizes.append(len(messages))
        self.last_batch_generation = {"seconds": 0.1, "batch_size": len(messages), "input_tokens": [10] * len(messages), "useful_input_tokens": 10 * len(messages), "padded_input_tokens": len(messages), "generated_tokens": [5] * len(messages), "total_generated_tokens": 5 * len(messages)}
        return [self.outputs.pop(0) for _ in messages]


class V2BenchmarkTests(unittest.TestCase):
    def test_story_run_treats_partial_as_completed_benchmark_observation(self):
        rows = [{"result": {"status": "ok"}}, {"result": {"status": "partial"}}]
        self.assertEqual(benchmark_cli.story_run_exit_code(rows, planned_cases=2), 0)
        self.assertEqual(benchmark_cli.story_run_exit_code(rows[:1], planned_cases=2), 2)
        self.assertEqual(benchmark_cli.story_run_exit_code([{"result": {"status": "error"}}], planned_cases=1), 2)

    def test_run_all_status_identifies_the_failed_track(self):
        self.assertTrue(hasattr(benchmark_cli, "summarize_run_status"))
        result = benchmark_cli.summarize_run_status(story_returncode=2, judge_returncode=0)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["tracks"], {"stories": "failed", "judge": "ok"})
        self.assertEqual(result["exit_code"], 2)

    def test_fixture_runner_uses_requested_batch_shape(self):
        llm = LLM([answer("consistent"), answer("contradiction"), answer("uncertain")])
        result = judge_fixture_items(items(), llm=llm, batch_size=2)
        self.assertEqual(llm.batch_sizes, [2, 1])
        self.assertEqual(result["metrics"]["facts_total"], 3)
        self.assertEqual(result["metrics"]["judge_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["main_batch_count"], 2)

    def test_accuracy_denominator_includes_invalid_outputs(self):
        predictions = [
            {"fixture_id": "F1", "status": "ok", "predicted_verdict": "consistent", "expected_verdict": "consistent"},
            {"fixture_id": "F2", "status": "error", "predicted_verdict": None, "expected_verdict": "contradiction"},
            {"fixture_id": "F3", "status": "ok", "predicted_verdict": "consistent", "expected_verdict": "uncertain"},
        ]
        score = score_judge_results(predictions)
        self.assertEqual(score["correct"], 1)
        self.assertEqual(score["facts_total"], 3)
        self.assertAlmostEqual(score["judge_accuracy"], 1 / 3)
        self.assertEqual(score["parse_failure_count"], 1)

    def test_oom_marks_configuration_without_lowering_batch(self):
        class OOM:
            device = "cuda"
            last_batch_generation = {}

            def _generate_batch(self, messages, max_new_tokens):
                raise RuntimeError("CUDA out of memory")

        result = judge_fixture_items(items(), llm=OOM(), batch_size=8)
        self.assertTrue(result["oom"])
        self.assertEqual(result["batch_size"], 8)
        self.assertEqual(result["metrics"]["facts_total"], 3)
        self.assertEqual(result["metrics"]["judge_accuracy"], 0.0)

    def test_markdown_compares_all_requested_batch_sizes(self):
        rows = []
        for batch_size in (1, 2, 4, 8):
            rows.append({"batch_size": batch_size, "oom": False, "metrics": {"peak_allocated_gib": 4.0, "peak_reserved_gib": 5.0, "total_judge_seconds": 10.0, "facts_per_second": 4.8, "judge_accuracy": 0.75}})
        markdown = benchmark_markdown(rows)
        for batch_size in (1, 2, 4, 8):
            self.assertIn(f"| {batch_size} |", markdown)

    def test_repeat_aggregation_uses_median_without_dropping_runs(self):
        runs = []
        for seconds, throughput in ((12.0, 4.0), (10.0, 4.8), (11.0, 4.4)):
            runs.append({"batch_size": 4, "oom": False, "status": "ok", "metrics": {"total_judge_seconds": seconds, "facts_per_second": throughput, "judge_accuracy": 0.75, "peak_allocated_gib": 4.2, "peak_reserved_gib": 5.0}})
        result = aggregate_repeats(runs)
        self.assertEqual(result["batch_size"], 4)
        self.assertEqual(result["repeat_count"], 3)
        self.assertEqual(result["metrics"]["total_judge_seconds"], 11.0)
        self.assertEqual(len(result["runs"]), 3)
        self.assertEqual(result["distributions"]["total_judge_seconds"]["values"], [12.0, 10.0, 11.0])
        self.assertAlmostEqual(result["distributions"]["total_judge_seconds"]["stddev"], (2 / 3) ** 0.5)

    def test_formal_judge_accepts_only_batch_eight(self):
        self.assertEqual(benchmark_cli.validate_formal_batch_sizes([8]), [8])
        with self.assertRaisesRegex(ValueError, "batch=8"):
            benchmark_cli.validate_formal_batch_sizes([1, 8])

    def test_cuda_runtime_metrics_report_total_and_incremental_peak(self):
        result = {"metrics": {}}
        attach_runtime_metrics(result, device="cuda", model_load_seconds=12.0, model_footprint_bytes=3 * 2**30, baseline_allocated_bytes=4 * 2**30, peak_allocated_bytes=5 * 2**30, peak_reserved_bytes=6 * 2**30)
        self.assertEqual(result["metrics"]["peak_allocated_gib"], 5.0)
        self.assertEqual(result["metrics"]["peak_reserved_gib"], 6.0)
        self.assertEqual(result["metrics"]["incremental_peak_allocated_gib"], 1.0)
        self.assertEqual(result["metrics"]["model_load_seconds"], 12.0)

    def test_story_quality_keeps_stage_denominators_separate(self):
        cases = [{"id": "S1", "gold_facts": [{"id": "G1", "expected_verdict": "矛盾", "acceptable_evidence_sets": [["C1"]]}]}]
        report = {
            "judge": {
                "events": [{"id": "E1"}],
                "items": [{"event_id": "E1", "status": "ok", "verdict": "contradiction"}],
                "retrieval": {"items": [{"event_id": "E1", "status": "ok", "evidence": [{"id": "C1"}]}]},
            }
        }
        score = score_story_quality(cases, [{"case_id": "S1", "result": report}], {"S1": {"G1": ["E1"]}}, k=5)
        self.assertEqual(score["recall_extraction"]["recall"], 1.0)
        self.assertEqual(score["recall_retrieval_at_k"]["recall"], 1.0)
        self.assertEqual(score["accuracy_judge"]["accuracy"], 1.0)
        self.assertEqual(score["end_to_end_conflict"]["f1"], 1.0)

    def test_validate_command_checks_inputs_without_loading_llm(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = benchmark_main(["validate"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["judge_fixture_items"], 48)
        self.assertEqual(result["story_cases"], 24)
        self.assertEqual(result["story_gold_facts"], 72)
        self.assertEqual(result["batch_sizes"], [8])
        self.assertEqual(result["repeats"], 3)


if __name__ == "__main__":
    unittest.main()
