import unittest
from llm_baseline import score_case

class BaselineScoringTests(unittest.TestCase):
    def test_missing_second_claim_is_not_a_pass(self):
        case={'claims':[{'quote':'雷拥有双心脏','verdict':'一致','kinds':['attribute']},{'quote':'雷能控制时间','verdict':'不确定','kinds':['attribute']}],'structures':[]}
        preview={'items':[{'proposed':{'kind':'attribute','text':'雷拥有双心脏'},'review':{'verdict':'一致','status':'ok','evidence':[]}}],'checks':[],'judgment':{'evidence':[]}}
        score=score_case(case,preview)
        self.assertEqual(score['verdict_correct'],1)
        self.assertEqual(score['claims_total'],2)
        self.assertFalse(score['complete'])

    def test_wrong_kind_and_verdict_are_separate_metrics(self):
        case={'claims':[{'quote':'灵能只能在月光下使用','verdict':'不确定','kinds':['rule']}],'structures':[]}
        preview={'items':[{'proposed':{'kind':'attribute','text':'灵能只能在月光下使用'},'review':{'verdict':'不确定','status':'ok','evidence':[]}}],'checks':[],'judgment':{'evidence':[]}}
        score=score_case(case,preview)
        self.assertEqual(score['verdict_correct'],1)
        self.assertEqual(score['kind_correct'],0)
        self.assertFalse(score['complete'])

    def test_false_citation_is_detected_independently(self):
        case={'claims':[{'quote':'雷左胸疼痛','verdict':'一致','kinds':['attribute'],'evidence_any':['左胸']}],'structures':[]}
        preview={'items':[{'proposed':{'kind':'attribute','text':'雷左胸疼痛'},'review':{'verdict':'一致','status':'ok','evidence':[{'chunk_id':'c1','quote':'不存在的证据'}]}}],'checks':[],'judgment':{'evidence':[{'id':'c1','text':'左胸疼痛'}]}}
        score=score_case(case,preview)
        self.assertEqual(score['citations_invalid'],1)
        self.assertFalse(score['complete'])

    def test_complete_requires_valid_preview_and_all_structures(self):
        case={'claims':[{'quote':'雷左胸疼痛','verdict':'一致','kinds':['attribute'],'evidence_any':['左胸']}],'structures':[{'kind':'timepoint','entity':'第十话'}]}
        fact={'proposed':{'kind':'attribute','text':'雷左胸疼痛'},'review':{'verdict':'一致','status':'ok','evidence':[{'chunk_id':'c1','quote':'左胸疼痛'}]}}
        evidence=[{'id':'c1','text':'左胸疼痛'}]
        preview={'status':'pending_confirmation','items':[fact], 'checks':[{'text':'雷左胸疼痛','result':{'evidence':evidence}}], 'judgment':{'evidence':evidence}}
        self.assertFalse(score_case(case,preview)['complete'])
        preview['items'].append({'proposed':{'kind':'timepoint','entity':'第十话'},'review':{'evidence':[]}})
        score=score_case(case,preview)
        self.assertTrue(score['complete'])
        self.assertEqual(score['retrieval_hits'],1)
        self.assertEqual(score['citation_core_hits'],1)
