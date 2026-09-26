import unittest

from agent_pipeline_v3.systems import build_baseline_system, build_candidate_system


class BenchmarkV3SystemTests(unittest.TestCase):
    def test_builders_pin_comparison_configuration(self):
        baseline = build_baseline_system(llm=object(), retriever=object(), dependencies={})
        candidate = build_candidate_system(llm=object(), retriever=object(), dependencies={})
        self.assertEqual(baseline.config.to_dict(), {
            "name": "baseline", "retrieval": "dense", "judge": "v1",
            "metadata_filter": False, "top_k": 5, "judge_batch_size": 8,
        })
        self.assertEqual(candidate.config.to_dict(), {
            "name": "candidate", "retrieval": "hybrid", "judge": "v2.1",
            "metadata_filter": False, "top_k": 5, "judge_batch_size": 8,
        })

    def test_baseline_delegates_dense_and_judge_v1(self):
        calls = []
        deps = {
            "extract": lambda text, **kw: calls.append(("extract", text)) or {"status": "ok"},
            "retrieve": lambda extraction, **kw: calls.append(("retrieve", kw["k"])) or {"status": "ok"},
            "judge": lambda retrieval, **kw: calls.append(("judge", kw["batch_size"])) or {"status": "ok"},
            "report": lambda judge: calls.append(("report",)) or {"status": "ok"},
        }
        system = build_baseline_system(llm="llm", retriever="dense", dependencies=deps)
        system.run_stages("story")
        self.assertEqual(calls, [("extract", "story"), ("retrieve", 5), ("judge", 8), ("report",)])

    def test_candidate_delegates_hybrid_without_metadata(self):
        calls = []
        deps = {
            "repair": lambda llm: ("repair", llm),
            "extract": lambda text, **kw: calls.append(("extract", kw["llm"])) or {"status": "ok"},
            "retrieve": lambda extraction, retriever, method, metadata_filter, k, progress=None: calls.append(("retrieve", method, metadata_filter, k)) or {"status": "ok"},
            "judge": lambda retrieval, llm, batch_size, progress=None: calls.append(("judge", batch_size)) or {"status": "ok"},
            "report": lambda judge: calls.append(("report",)) or {"status": "ok"},
        }
        system = build_candidate_system(llm="llm", retriever="hybrid", dependencies=deps)
        system.run_stages("story")
        self.assertEqual(calls, [("extract", ("repair", "llm")), ("retrieve", "hybrid", False, 5), ("judge", 8), ("report",)])


if __name__ == "__main__":
    unittest.main()
