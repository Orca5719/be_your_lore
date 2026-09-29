import json
import unittest

from agent_pipeline_v2.extractor import split_spans
from agent_pipeline_v2.judge import judge_events, validate_judge


def retrieval_report(count=5, with_evidence=True):
    text = "雷拥有两颗心脏。"
    spans = split_spans(text)
    events = []
    items = []
    for number in range(1, count + 1):
        event = {
            "id": f"E{number}",
            "actors": ["雷"],
            "event": f"雷拥有两颗心脏（核对项{number}）",
            "mental_state": None,
            "explicit": True,
            "modality": "observed",
            "conditions": [],
            "source_ids": ["S1"],
            "context_ids": [],
            "check_reason": "mechanism",
        }
        events.append(event)
        evidence = [{
            "id": f"C{number}",
            "text": "雷拥有两颗心脏。",
            "file": "characters.md",
            "start_line": 6,
            "end_line": 6,
            "heading_path": ["雷", "生理结构"],
            "score": 0.9,
        }] if with_evidence else []
        items.append({"event_id": event["id"], "event": event, "query": event["event"], "status": "ok", "evidence": evidence})
    extraction = {
        "schema_version": "agent-pipeline-v2-extraction-v1",
        "stage": "extraction",
        "status": "ok",
        "text": text,
        "spans": spans,
        "events": events,
        "ignored_spans": [],
        "non_event_span_ids": [],
        "uncovered_span_ids": [],
        "rejected": [],
    }
    return {
        "schema_version": "agent-pipeline-v2-retrieval-v1",
        "stage": "retrieval",
        "status": "ok",
        "extraction": extraction,
        "events": events,
        "items": items,
        "top_k": 5,
    }


class LLM:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self):
        self.batch_sizes = []
        self.last_batch_generation = {}

    def _generate_batch(self, message_batches, max_new_tokens):
        self.batch_sizes.append(len(message_batches))
        self.last_batch_generation = {
            "seconds": 0.2,
            "batch_size": len(message_batches),
            "input_tokens": [100] * len(message_batches),
            "useful_input_tokens": 100 * len(message_batches),
            "padded_input_tokens": 10 * len(message_batches),
            "generated_tokens": [20] * len(message_batches),
            "total_generated_tokens": 20 * len(message_batches),
        }
        return [json.dumps({
            "verdict": "consistent",
            "citations": [{"evidence_id": "L1", "quote": "雷拥有两颗心脏。"}],
            "reason": "设定直接支持",
            "assessment": {"same_subject": True, "evidence_applicable": True, "relation": "direct_support", "assumptions": []},
        }, ensure_ascii=False) for _ in message_batches]


class V2JudgeTests(unittest.TestCase):
    def test_five_facts_run_as_two_two_one(self):
        llm = LLM()
        result = judge_events(retrieval_report(5), batch_size=2, llm=llm)
        self.assertEqual(llm.batch_sizes, [2, 2, 1])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["items"]), 5)
        self.assertEqual(result["metrics"]["main_batch_count"], 3)
        self.assertEqual(result["metrics"]["useful_input_tokens"], 500)
        self.assertEqual(result["metrics"]["padded_input_tokens"], 50)
        validate_judge(result)

    def test_no_evidence_is_uncertain_without_model_generation(self):
        llm = LLM()
        result = judge_events(retrieval_report(1, with_evidence=False), batch_size=8, llm=llm)
        self.assertEqual(llm.batch_sizes, [])
        self.assertEqual(result["items"][0]["verdict"], "uncertain")
        self.assertEqual(result["items"][0]["origin"], "program_no_evidence")

    def test_batch_size_must_be_positive_integer(self):
        with self.assertRaisesRegex(ValueError, "正整数"):
            judge_events(retrieval_report(1), batch_size=True, llm=LLM())

    def test_citations_are_mapped_back_to_real_chunk_ids(self):
        result = judge_events(retrieval_report(1), batch_size=1, llm=LLM())
        self.assertEqual(result["items"][0]["citations"][0]["chunk_id"], "C1")


if __name__ == "__main__":
    unittest.main()
