import json
import unittest

from agent_pipeline_v2.extractor import extract_events, split_spans, validate_extraction


def event(source_id="S1"):
    return {
        "id": "E1",
        "actors": ["雷"],
        "event": "雷拥有两颗心脏",
        "mental_state": None,
        "explicit": True,
        "modality": "observed",
        "conditions": [],
        "source_ids": [source_id],
        "context_ids": [],
        "check_reason": "mechanism",
    }


def report():
    text = "雷拥有两颗心脏。他喝了一口水。随后，"
    spans = split_spans(text)
    return {
        "schema_version": "agent-pipeline-v2-extraction-v1",
        "stage": "extraction",
        "status": "ok",
        "text": text,
        "spans": spans,
        "events": [event("S1")],
        "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
        "non_event_span_ids": ["S3"],
        "rejected": [],
    }


class V2ExtractorContractTests(unittest.TestCase):
    def test_every_span_has_exactly_one_disposition(self):
        rows = validate_extraction(report())
        self.assertEqual(set(rows), {"S1", "S2", "S3"})
        self.assertEqual(rows["S1"]["disposition"], "event")
        self.assertEqual(rows["S2"]["disposition"], "ignored")
        self.assertEqual(rows["S3"]["disposition"], "non_event")

    def test_missing_span_is_rejected(self):
        value = report()
        value["non_event_span_ids"] = []
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            validate_extraction(value)

    def test_overlapping_dispositions_are_rejected(self):
        value = report()
        value["ignored_spans"].append({"source_id": "S1", "reason": "routine"})
        with self.assertRaisesRegex(ValueError, "重复处置"):
            validate_extraction(value)

    def test_invalid_ignore_reason_is_rejected(self):
        value = report()
        value["ignored_spans"][0]["reason"] = "mechanism"
        with self.assertRaisesRegex(ValueError, "忽略原因"):
            validate_extraction(value)

    def test_event_actor_must_appear_in_its_sources_or_context(self):
        value = report()
        value["events"][0]["actors"] = ["亚巴顿"]
        with self.assertRaisesRegex(ValueError, "主体缺少原文依据"):
            validate_extraction(value)

    def test_empty_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "输入不能为空"):
            split_spans("  \n")


class FakeLLM:
    device = "cpu"
    load_seconds = 0.0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.last_generation = {}

    def _generate(self, messages, max_new_tokens):
        self.calls.append((messages, max_new_tokens))
        self.last_generation = {"seconds": 0.01, "input_tokens": 20, "generated_tokens": 10}
        return self.outputs.pop(0)


class V2ExtractorOrchestrationTests(unittest.TestCase):
    def test_one_stage_returns_only_checkable_events_and_audits_other_spans(self):
        answer = {
            "events": [
                {
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
            ],
            "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
            "non_event_span_ids": ["S3"],
        }
        llm = FakeLLM([json.dumps(answer, ensure_ascii=False)])
        result = extract_events("雷拥有两颗心脏。他喝了一口水。随后，", llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["id"] for row in result["events"]], ["E1"])
        self.assertEqual(result["ignored_spans"], answer["ignored_spans"])
        self.assertEqual(len(llm.calls), 1)
        validate_extraction(result)

    def test_invalid_first_response_is_retried_in_the_same_stage(self):
        valid = {
            "events": [],
            "ignored_spans": [{"source_id": "S1", "reason": "routine"}],
            "non_event_span_ids": [],
        }
        llm = FakeLLM(["not json", json.dumps(valid, ensure_ascii=False)])
        result = extract_events("雷喝了一口水。", llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("上次回复不合格", llm.calls[1][0][-1]["content"])

    def test_two_windows_still_use_the_same_extractor_stage(self):
        text = "雷喝了一口水。" * 12
        spans = split_spans(text)
        first_ids = list(spans)[:8]
        remaining_ids = list(spans)[8:]
        outputs = []
        for ids in (first_ids, remaining_ids):
            outputs.append(json.dumps({
                "events": [],
                "ignored_spans": [{"source_id": sid, "reason": "routine"} for sid in ids],
                "non_event_span_ids": [],
            }, ensure_ascii=False))
        llm = FakeLLM(outputs)
        result = extract_events(text, llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual({call["purpose"] for call in result["calls"]}, {"extract_checkable_events"})
        self.assertEqual(len(result["calls"]), 2)

    def test_resolves_actor_evidence_from_nearest_preceding_span(self):
        answer = {
            "events": [{
                "actors": ["雷"],
                "event": "雷喝了一口水",
                "mental_state": None,
                "explicit": True,
                "modality": "observed",
                "conditions": [],
                "source_ids": ["S2"],
                "context_ids": [],
                "check_reason": "mechanism",
            }],
            "ignored_spans": [{"source_id": "S1", "reason": "routine"}],
            "non_event_span_ids": [],
        }
        llm = FakeLLM([json.dumps(answer, ensure_ascii=False)])
        result = extract_events("雷坐在桌边，他喝了一口水。", llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["events"][0]["context_ids"], ["S1"])
        self.assertIn("actor_context_added", result["calls"][0]["wire_normalizations"])

    def test_normalizes_scalar_actor_list_without_retry(self):
        answer = {
            "events": [{
                "actors": "雷",
                "event": "雷拥有让大厅时间暂停的能力",
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
        }
        llm = FakeLLM([json.dumps(answer, ensure_ascii=False)])
        result = extract_events("雷拥有让大厅时间暂停的能力。", llm=llm)
        self.assertEqual(result["events"][0]["actors"], ["雷"])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("scalar_actors_to_list", result["calls"][0]["wire_normalizations"])

    def test_prefers_ignored_disposition_over_duplicate_non_event(self):
        answer = {
            "events": [],
            "ignored_spans": [{"source_id": "S1", "reason": "routine"}],
            "non_event_span_ids": ["S1"],
        }
        llm = FakeLLM([json.dumps(answer, ensure_ascii=False)])
        result = extract_events("雷喝了一口水。", llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ignored_spans"], answer["ignored_spans"])
        self.assertEqual(result["non_event_span_ids"], [])
        self.assertIn("duplicate_non_event_removed", result["calls"][0]["wire_normalizations"])

    def test_drops_dispositions_for_context_only_spans(self):
        text = "雷喝了一口水。" * 9
        spans = split_spans(text)
        first_ids = list(spans)[:8]
        last_id = list(spans)[8]
        first = {
            "events": [],
            "ignored_spans": [{"source_id": sid, "reason": "routine"} for sid in first_ids],
            "non_event_span_ids": [],
        }
        second = {
            "events": [],
            "ignored_spans": [{"source_id": last_id, "reason": "routine"}],
            "non_event_span_ids": [first_ids[-1]],
        }
        llm = FakeLLM([json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False)])
        result = extract_events(text, llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["non_event_span_ids"], [])
        self.assertIn("context_disposition_removed", result["calls"][1]["wire_normalizations"])

    def test_missing_targets_are_recovered_without_discarding_valid_rows(self):
        first = {
            "events": [],
            "ignored_spans": [{"source_id": "S1", "reason": "routine"}],
            "non_event_span_ids": [],
        }
        recovery = {
            "events": [],
            "ignored_spans": [{"source_id": "S2", "reason": "routine"}],
            "non_event_span_ids": [],
        }
        llm = FakeLLM([json.dumps(first, ensure_ascii=False), json.dumps(recovery, ensure_ascii=False)])
        result = extract_events("雷坐在桌边，他喝了一口水。", llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["source_id"] for row in result["ignored_spans"]], ["S1", "S2"])
        self.assertEqual(result["calls"][0]["recovered_target_ids"], ["S2"])
        self.assertEqual(len(llm.calls), 2)

    def test_context_only_repeated_event_is_removed_from_a_window(self):
        text = "雷喝了一口水。" * 9
        spans = split_spans(text)
        first_ids = list(spans)[:8]
        last_id = list(spans)[8]
        first = {
            "events": [],
            "ignored_spans": [{"source_id": sid, "reason": "routine"} for sid in first_ids],
            "non_event_span_ids": [],
        }
        second = {
            "events": [{
                "actors": ["雷"],
                "event": "雷喝了一口水",
                "mental_state": None,
                "explicit": True,
                "modality": "observed",
                "conditions": [],
                "source_ids": [first_ids[-1]],
                "context_ids": [],
                "check_reason": "mechanism",
            }],
            "ignored_spans": [{"source_id": last_id, "reason": "routine"}],
            "non_event_span_ids": [],
        }
        llm = FakeLLM([json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False)])
        result = extract_events(text, llm=llm)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["events"], [])
        self.assertIn("context_event_removed", result["calls"][1]["wire_normalizations"])


if __name__ == "__main__":
    unittest.main()
