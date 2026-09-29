import unittest
from story_retrieval import retrieve_facts, fact_query


def fact(id,subject,predicate,obj=None,context=''):
    return dict(id=id,subject=subject,predicate=predicate,object=obj,
                source_text=subject+predicate+(obj or ''),context_text=context,time=None)


class StoryRetrievalTests(unittest.TestCase):
    def test_queries_and_evidence_are_isolated(self):
        class Fake:
            def __init__(self):self.calls=[]
            def search(self,query,k):
                self.calls.append((query,k))
                return [dict(id='C'+str(len(self.calls)),text='证据',file='test.md',start_line=1,end_line=1,heading_path=['测试'],score=0.8)]
        first=fact('F1','雷','左胸寄宿','亚巴顿')
        second=fact('F4','德尔塔','知道','雷的秘密')
        extraction=dict(status='partial',story='整段故事包含银河旅游指南',facts=[first,second],
                        pending_facts=[fact('F3','雷','喝水')],rejected=[{'id':'F2'}],discarded=[{'id':'F5'}])
        retriever=Fake()
        result=retrieve_facts(extraction,retriever,k=3)
        self.assertEqual([x['fact']['id'] for x in result['items']],['F1','F4'])
        self.assertEqual(result['items'][0]['evidence'][0]['id'],'C1')
        self.assertEqual(result['items'][1]['evidence'][0]['id'],'C2')
        self.assertEqual(len(retriever.calls),2)
        self.assertNotIn('德尔塔',retriever.calls[0][0])
        self.assertNotIn('银河旅游指南',retriever.calls[0][0])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(extraction['facts'],[first,second])

    def test_no_facts_does_not_search(self):
        class Fake:
            def search(self,*args,**kwargs):raise AssertionError('不应检索')
        result=retrieve_facts(dict(status='ok',facts=[]),Fake())
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['items'],[])

    def test_failure_is_not_empty_success_and_other_facts_continue(self):
        class Fake:
            def search(self,query,k):
                if '雷' in query:raise ValueError('完整编码输入超过512 tokens')
                return []
        result=retrieve_facts(dict(status='ok',facts=[fact('F1','雷','失明'),fact('F2','月城','失明')]),Fake())
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['items'][0]['status'],'error')
        self.assertIn('512',result['items'][0]['error'])
        self.assertEqual(result['items'][1]['status'],'ok')
        self.assertEqual(result['items'][1]['evidence'],[])

    def test_query_preserves_conditions_and_null_object(self):
        query=fact_query(fact('F1','雷','右胸温暖',context='拉古艾尔施展治疗时，'))
        self.assertIn('雷',query)
        self.assertIn('右胸温暖',query)
        self.assertIn('治疗时',query)
        self.assertNotIn('None',query)

    def test_invalid_k_and_extraction_error(self):
        for k in (0,True,1.2):
            with self.assertRaises(ValueError):retrieve_facts(dict(status='ok',facts=[]),None,k)
        result=retrieve_facts(dict(status='error',facts=[],error='JSON坏'),None)
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['items'],[])
