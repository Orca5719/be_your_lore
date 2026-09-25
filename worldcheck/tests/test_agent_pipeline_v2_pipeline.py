import json
import unittest

from agent_pipeline_v2.pipeline import process_story


class Retriever:
    def search(self, query, k=5):
        return [{
            "id": "C1",
            "text": "雷拥有两颗心脏。",
            "file": "characters.md",
            "start_line": 6,
            "end_line": 6,
            "heading_path": ["雷", "生理结构"],
            "score": 0.9,
        }]


class LLM:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self, extraction="valid"):
        self.extraction = extraction
        self.last_generation = {}
        self.last_batch_generation = {}

    def _generate(self, messages, max_new_tokens):
        self.last_generation = {"seconds": 0.1}
        if self.extraction == "invalid":
            return "not json"
        return json.dumps({
            "events": [{
                "actors": ["雷"],
                "event": "雷拥有两颗心脏",
                "mental_state": None,
                "explicit": True,
                "modality": "observed",
                "conditions": [],
                "source_ids": ["S1"],
                "context_ids": [],
                "check_reason": "mechanism",
            }],
            "ignored_spans": [],
            "non_event_span_ids": [],
        }, ensure_ascii=False)

    def _generate_batch(self, messages, max_new_tokens):
        self.last_batch_generation = {"seconds": 0.2, "batch_size": len(messages), "input_tokens": [100] * len(messages), "useful_input_tokens": 100 * len(messages), "padded_input_tokens": 0, "generated_tokens": [20] * len(messages), "total_generated_tokens": 20 * len(messages)}
        return [json.dumps({
            "verdict": "consistent",
            "citations": [{"evidence_id": "L1", "quote": "雷拥有两颗心脏。"}],
            "reason": "设定直接支持",
            "assessment": {"same_subject": True, "evidence_applicable": True, "relation": "direct_support", "assumptions": []},
        }, ensure_ascii=False) for _ in messages]


class V2PipelineTests(unittest.TestCase):
    def test_story_flows_through_extraction_retrieval_and_batched_judge(self):
        result = process_story("雷拥有两颗心脏。", llm=LLM(), retriever=Retriever(), judge_batch_size=4)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["counts"]["consistent"], 1)
        self.assertEqual(result["findings"][0]["event"], "雷拥有两颗心脏")
        self.assertEqual(result["judge"]["batch_size"], 4)

    def test_extraction_failure_propagates_to_final_report(self):
        result = process_story("雷拥有两颗心脏。", llm=LLM("invalid"), retriever=Retriever(), judge_batch_size=1)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["processing_complete"])
        self.assertEqual(result["summary"]["checked_event_count"], 0)


if __name__ == "__main__":
    unittest.main()
