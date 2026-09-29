import unittest

from agent_pipeline_v2.extractor import split_spans
from agent_pipeline_v2.retrieval import event_query, retrieve_events, validate_retrieval


def extraction(status="ok"):
    text = "雷拥有两颗心脏。他喝了一口水。"
    spans = split_spans(text)
    event = {
        "id": "E1",
        "actors": ["雷"],
        "event": "雷拥有两颗心脏",
        "mental_state": None,
        "explicit": True,
        "modality": "observed",
        "conditions": [],
        "source_ids": ["S1"],
        "context_ids": [],
        "check_reason": "mechanism",
    }
    return {
        "schema_version": "agent-pipeline-v2-extraction-v1",
        "stage": "extraction",
        "status": status,
        "text": text,
        "spans": spans,
        "events": [event],
        "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
        "non_event_span_ids": [],
        "uncovered_span_ids": [],
        "rejected": [],
    }


class Retriever:
    def __init__(self, fail=False):
        self.fail = fail
        self.queries = []

    def search(self, query, k=5):
        self.queries.append((query, k))
        if self.fail:
            raise ValueError("index broken")
        return [
            {
                "id": "C1",
                "text": "雷拥有两颗心脏。",
                "file": "characters.md",
                "start_line": 6,
                "end_line": 6,
                "heading_path": ["雷", "生理结构"],
                "score": 0.9,
            }
        ]


class V2RetrievalTests(unittest.TestCase):
    def test_query_uses_event_and_original_source(self):
        report = extraction()
        query = event_query(report["events"][0], report["spans"])
        self.assertIn("事件：雷拥有两颗心脏", query)
        self.assertIn("局部原文：雷拥有两颗心脏。", query)

    def test_retrieves_directly_from_v2_extraction(self):
        backend = Retriever()
        result = retrieve_events(extraction(), retriever=backend, k=5)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["items"][0]["evidence"][0]["id"], "C1")
        self.assertEqual(backend.queries[0][1], 5)
        validate_retrieval(result)

    def test_event_failure_is_preserved(self):
        result = retrieve_events(extraction(), retriever=Retriever(fail=True), k=5)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["items"][0]["status"], "error")
        self.assertIn("index broken", result["items"][0]["error"])

    def test_top_k_must_be_positive_integer(self):
        with self.assertRaisesRegex(ValueError, "正整数"):
            retrieve_events(extraction(), retriever=Retriever(), k=True)


if __name__ == "__main__":
    unittest.main()
