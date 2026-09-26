import unittest
from world_benchmark import score_case

class WorldBenchmarkTests(unittest.TestCase):
    def preview(self,items):
        return {'status':'pending_confirmation','items':items,'checks':[],'judgment':{'evidence':[]}}
    def item(self,text,verdict,entity='雷',kind='attribute',time=None):
        return {'proposed':{'text':text,'kind':kind,'entity':entity,'time':time},'review':{'status':'ok','verdict':verdict,'evidence':[]}}
    def case(self,claims,overall='不确定'):
        return {'claims':claims,'expected_overall':overall,'required_structures':[]}
    def claim(self,q,v,**kw):
        return dict(input_quote=q,expected_verdict=v,allowed_kinds=['attribute'],evidence=[],**kw)
    def test_mixed_missing_claim_is_not_complete(self):
        case=self.case([self.claim('雷有两颗心脏','一致'),self.claim('雷能控制气温','不确定')])
        score=score_case(case,self.preview([self.item('雷有两颗心脏','一致')]))
        self.assertEqual(score['claims_total'],2)
        self.assertEqual(score['verdict_correct'],1)
        self.assertFalse(score['complete'])
        self.assertFalse(score['overall_correct'])
    def test_foreign_entity_definition_is_not_factual_consistency(self):
        case=self.case([self.claim('艾琳是校长','不确定')])
        score=score_case(case,self.preview([self.item('艾琳','不确定','艾琳','entity')]))
        self.assertEqual(score['observed_overall'],'缺失/失败')
        self.assertEqual(score['claims_found'],0)
    def test_context_and_subject_are_separate_from_verdict(self):
        case=self.case([self.claim('德尔塔知道身份','一致',context_terms=['第三话','之后'],subject_any=['德尔塔'])],'一致')
        score=score_case(case,self.preview([self.item('德尔塔知道身份','一致','雷',time='第三话')]))
        self.assertTrue(score['overall_correct'])
        self.assertEqual(score['verdict_correct'],1)
        self.assertEqual(score['context_correct'],0)
        self.assertEqual(score['subject_correct'],0)
        self.assertFalse(score['complete'])
    def test_complete_and_unmatched_extra_conflict(self):
        case=self.case([self.claim('雷有两颗心脏','一致')],'一致')
        preview=self.preview([self.item('雷有两颗心脏','一致')])
        self.assertTrue(score_case(case,preview)['complete'])
        preview['items'].append(self.item('其他事实','矛盾'))
        score=score_case(case,preview)
        self.assertFalse(score['complete'])
        self.assertFalse(score['overall_correct'])
        self.assertEqual(score['unmatched_facts'],1)
    def test_invalid_citation_blocks_complete(self):
        case=self.case([self.claim('雷有两颗心脏','一致')],'一致')
        preview=self.preview([self.item('雷有两颗心脏','一致')])
        preview['items'][0]['review']['evidence']=[{'chunk_id':'不存在','quote':'伪证'}]
        score=score_case(case,preview)
        self.assertEqual(score['citations_invalid'],1)
        self.assertFalse(score['complete'])

    def test_annotation_cannot_merge_condition_with_fact_across_comma(self):
        import json
        from world_benchmark import ROOT,validate_dataset
        data=json.loads((ROOT/'evaluation/world_benchmark_160_v1.json').read_text(encoding='utf-8'))
        validate_dataset(data,ROOT/'lore')
        case=next(c for c in data['cases'] if c['id']=='wb007')
        case['claims'][0]['input_quote']=case['text'].rstrip('。')
        with self.assertRaisesRegex(ValueError,'跨越分句'):
            validate_dataset(data,ROOT/'lore')
