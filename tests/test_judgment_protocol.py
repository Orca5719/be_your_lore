import copy
import json
import unittest
from judgment_protocol import validate_answer

class JudgmentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.text = '雷的左心脏寄宿亚巴顿。'
        self.evidence = {'manual-01': self.text}
        self.answer = {'findings': [{'input_quote': self.text, 'verdict': '一致', 'evidence': [{'chunk_id': 'manual-01', 'quote': self.text}], 'reason': '证据直接支持。', 'new_candidates': []}]}

    def test_support_and_program_aggregation(self):
        self.assertEqual(validate_answer(json.dumps(self.answer), self.text, self.evidence)['overall_verdict'], '一致')
        unknown = copy.deepcopy(self.answer['findings'][0])
        unknown.update(verdict='不确定', evidence=[], new_candidates=[self.text])
        self.answer['findings'].append(unknown)
        self.assertEqual(validate_answer(json.dumps(self.answer), self.text, self.evidence)['overall_verdict'], '不确定')
        unknown.update(verdict='矛盾', evidence=self.answer['findings'][0]['evidence'], new_candidates=[])
        self.assertEqual(validate_answer(json.dumps(self.answer), self.text, self.evidence)['overall_verdict'], '矛盾')

    def test_fake_evidence_and_missing_evidence_rejected(self):
        item = self.answer['findings'][0]
        item['evidence'][0]['quote'] = '雷有三个心脏'
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(self.answer), self.text, self.evidence)
        item['evidence'] = []
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(self.answer), self.text, self.evidence)

    def test_invalid_output_and_input_quote_rejected(self):
        for raw in ['not json', '{}', '{"findings": []}']:
            with self.assertRaises(ValueError):
                validate_answer(raw, self.text, self.evidence)
        self.answer['findings'][0]['input_quote'] = '编造的剧情'
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(self.answer), self.text, self.evidence)

    def test_candidates_must_be_exact_finding_and_uncertain(self):
        item = self.answer['findings'][0]
        item.update(verdict='不确定', evidence=[])
        for candidates in [['时间操纵与心脏的关系'], ['雷是否有双心脏'], [self.text, self.text]]:
            item['new_candidates'] = candidates
            with self.assertRaises(ValueError):
                validate_answer(json.dumps(self.answer), self.text, self.evidence)
        item['new_candidates'] = [self.text]
        self.assertEqual(validate_answer(json.dumps(self.answer), self.text, self.evidence)['overall_verdict'], '不确定')
        item.update(verdict='一致', evidence=[{'chunk_id': 'manual-01', 'quote': self.text}])
        with self.assertRaises(ValueError):
            validate_answer(json.dumps(self.answer), self.text, self.evidence)

    def test_uncertain_input_kept_for_review_even_if_model_omits_candidate(self):
        item = self.answer['findings'][0]
        item.update(verdict='不确定', evidence=[], new_candidates=[])
        result = validate_answer(json.dumps(self.answer), self.text, self.evidence)
        self.assertEqual(result['review_candidates'], [self.text])
