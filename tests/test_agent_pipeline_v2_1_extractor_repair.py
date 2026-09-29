import json


class FakeLLM:
    device = "cpu"
    load_seconds = 0
    last_generation = {}

    def __init__(self, value):
        self.value = value

    def _generate(self, messages, max_new_tokens=1536):
        return json.dumps(self.value, ensure_ascii=False)


def messages():
    payload = {
        "target_spans": {"S1": "故事发生在房间里。", "S2": "旧教堂封印可以由一个人独自打开。", "S3": "雷在第三话说明秘密。"},
        "context_spans": {},
    }
    return [{"role": "system", "content": "x"}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def test_wire_repair_fixes_unambiguous_field_aliases_and_enums():
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM

    value = {"events": [{"actors": ["一个人"], "event": "封印可由一个人打开", "mentalological": None, "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S2"], "contextips": [], "checkreason": "mechanibility"}], "ignored_spans": [], "non_event_span_ids": ["S1", "S3"]}
    repaired = json.loads(ExtractionRepairLLM(FakeLLM(value))._generate(messages()))
    event = repaired["events"][0]
    assert set(event) == {"actors", "event", "mental_state", "explicit", "modality", "conditions", "source_ids", "context_ids", "check_reason"}
    assert event["check_reason"] == "mechanism"


def test_wire_repair_replaces_only_explicit_placeholder_actor():
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM

    value = {"events": [{"actors": ["未知人物"], "event": "旧教堂封印可以由一个人独自打开", "mental_state": None, "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S2"], "context_ids": [], "check_reason": "mechanism"}], "ignored_spans": [], "non_event_span_ids": ["S1", "S3"]}
    repaired = json.loads(ExtractionRepairLLM(FakeLLM(value))._generate(messages()))
    assert repaired["events"][0]["actors"] == ["一个人"]


def test_wire_repair_drops_ungrounded_event_and_returns_span_for_recovery():
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM

    value = {"events": [{"actors": ["雷"], "event": "雷拥有两颗心脏", "mental_state": None, "explicit": True, "modality": "observed", "conditions": [], "source_ids": ["S1"], "context_ids": [], "check_reason": "mechanism"}], "ignored_spans": [], "non_event_span_ids": []}
    repaired = json.loads(ExtractionRepairLLM(FakeLLM(value))._generate(messages()))
    assert repaired["events"] == []
    assert repaired["non_event_span_ids"] == []
    assert set(repaired) == {"events", "ignored_spans", "non_event_span_ids"}
