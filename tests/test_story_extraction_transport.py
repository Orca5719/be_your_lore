import json
import unittest
from story_extraction import extract_story


class FakeJudge:
    device = 'cpu'
    last_generation = {'seconds': 1}

    def __init__(self, raw='{"facts":[]}', tokenizer=None):
        self.raw = raw
        self.messages = []
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def _generate(self, messages, max_new_tokens):
        self.messages.append((messages, max_new_tokens))
        if len(self.messages) == 1:
            return self.raw
        return '{"decision":"keep","reason":"明确生理特征"}'


class TransportTests(unittest.TestCase):
    def test_narrative_and_compact_references_keep_offsets_local(self):
        judge = FakeJudge()
        text = '雷站起身。雷有两颗心脏。'
        result = extract_story(text, judge=judge)
        payload = json.loads(judge.messages[0][0][1]['content'])
        self.assertEqual(payload['story'],text)
        self.assertEqual(payload['source_spans'], {'S1': '雷站起身。', 'S2': '雷有两颗心脏。'})
        self.assertEqual(result['story'], text)
        self.assertEqual(result['unselected_spans']['S2']['start'], 5)

    def test_output_reservation_uses_actual_tokenized_request(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return list(range(2800))
        judge = FakeJudge(tokenizer=Tokenizer())
        result = extract_story('雷站起身。', judge=judge)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(judge.messages[0][1], 1296)
        self.assertEqual(result['generation_budget']['input_tokens'], 2800)

    def test_complete_json_with_explanation_keeps_raw_and_validates_facts(self):
        row = {'subject':'雷', 'predicate':'有两颗心脏', 'type':'character_attribute', 'source_ids':['S1']}
        raw = json.dumps({'facts':[row]}, ensure_ascii=False) + '\n\n说明：这是身体特征。'
        result = extract_story('雷有两颗心脏。', judge=FakeJudge(raw))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['raw_output'], raw)
        self.assertEqual(result['format_normalizations'], ['trailing_explanation_removed'])
        self.assertEqual(result['facts'][0]['source_text'], '雷有两颗心脏。')

    def test_second_json_and_incomplete_json_are_not_repaired(self):
        for raw in ['{"facts":[]}\n{"facts":[]}', '{"facts":[', '{"facts":[]}\n说明：\n{"facts":[]}']:
            result = extract_story('雷站起身。', judge=FakeJudge(raw))
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['facts'], [])
            self.assertEqual(result['raw_output'], raw)

    def test_single_trailing_quote_is_logged_without_changing_object(self):
        result = extract_story('雷站起身。', judge=FakeJudge('{"facts":[]}"'))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['format_normalizations'], ['single_trailing_quote_removed'])

    def test_insufficient_budget_does_not_call_model_or_truncate(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return list(range(3900))
        judge=FakeJudge(tokenizer=Tokenizer())
        result=extract_story('雷有两颗心脏。',judge=judge)
        self.assertEqual(result['status'],'error')
        self.assertEqual(judge.messages,[])
        self.assertIn('未截断原文',result['error'])

    def test_explanation_does_not_bypass_reference_validation(self):
        raw = '{"facts":[{"subject":"雷","predicate":"有两颗心脏","type":"character_attribute","source_ids":["S99"]}]}\n说明：身体特征'
        result = extract_story('雷有两颗心脏。', judge=FakeJudge(raw))
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['facts'], [])
        self.assertEqual(len(result['rejected']), 1)
