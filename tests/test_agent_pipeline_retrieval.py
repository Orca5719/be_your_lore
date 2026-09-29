import copy
import unittest
from agent_pipeline.retrieval import retrieve_events,event_query
from test_agent_pipeline_filtering import upstream,FakeLLM,decision
from agent_pipeline import filter_events


def filtered(count=2,status='ok'):
    return filter_events(upstream(count,status),llm=FakeLLM([{'decisions':[decision(f'E{i}') for i in range(1,count+1)]}]) if count else None)


class FakeRetriever:
    def __init__(self):self.queries=[]
    def search(self,query,k=5):
        self.queries.append((query,k))
        if '失败' in query:raise ValueError('完整编码输入超过512 tokens')
        return [dict(id='C1',text='设定原文',file='characters.md',start_line=6,end_line=6,title_path=['雷'],score=0.7)]


class EventRetrievalTests(unittest.TestCase):
    def test_kept_events_retrieve_without_mutating_upstream(self):
        report=filtered();before=copy.deepcopy(report);retriever=FakeRetriever()
        result=retrieve_events(report,retriever=retriever,k=5)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['items']),2)
        self.assertEqual(result['items'][0]['evidence'][0]['file'],'characters.md')
        self.assertEqual(report,before)

    def test_query_is_local_and_preserves_negation_and_modality(self):
        event=upstream(1)['events'][0]
        event.update(event='声称没有见过德尔塔',modality='speech',conditions=['地震前'],contexts=[{'text':'雷来到实验室。'}])
        query=event_query(event)
        self.assertIn('声称没有见过德尔塔',query)
        self.assertIn('speech',query)
        self.assertIn('地震前',query)
        self.assertIn('雷来到实验室',query)

    def test_support_event_is_preserved_without_independent_search(self):
        report=filtered();report['decisions'][0].update(support_only=True,required_by_event_ids=['E2'])
        retriever=FakeRetriever();result=retrieve_events(report,retriever=retriever)
        self.assertEqual(len(retriever.queries),1)
        self.assertEqual(result['items'][0]['status'],'support_only')
        self.assertEqual(result['items'][0]['required_by_event_ids'],['E2'])

    def test_pending_and_ignored_are_not_searched(self):
        report=filter_events(upstream(2),llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2','review')]}]))
        retriever=FakeRetriever();result=retrieve_events(report,retriever=retriever)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(retriever.queries,[])
        self.assertEqual(result['pending_event_ids'],['E2'])

    def test_empty_selection_does_not_load_embedding_or_index(self):
        result=retrieve_events(filtered(0),index_directory='missing_index')
        self.assertEqual(result['status'],'ok')
        self.assertFalse(result['encoder_loaded_this_request'])

    def test_one_failed_query_does_not_erase_other_results(self):
        report=filtered();report['events'][0]['event']='失败'
        result=retrieve_events(report,retriever=FakeRetriever())
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['items'][0]['status'],'error')
        self.assertEqual(result['items'][1]['status'],'ok')

    def test_upstream_error_is_not_masked_or_searched(self):
        report=filtered(0,'error');retriever=FakeRetriever()
        result=retrieve_events(report,retriever=retriever)
        self.assertEqual(result['status'],'error')
        self.assertEqual(retriever.queries,[])

    def test_invalid_k_and_partition_rejected_before_loading(self):
        with self.assertRaises(ValueError):retrieve_events(filtered(),k=True)
        report=filtered();report['selected_events']=[]
        with self.assertRaises(ValueError):retrieve_events(report)

    def test_identical_queries_cache_evidence_but_preserve_event_ids(self):
        report=filtered();report['events'][1].update({key:value for key,value in report['events'][0].items() if key!='id'})
        retriever=FakeRetriever();result=retrieve_events(report,retriever=retriever)
        self.assertEqual(len(retriever.queries),1)
        self.assertEqual([x['event_id'] for x in result['items']],['E1','E2'])
        self.assertTrue(result['items'][1]['cache_hit'])

    def test_quotes_preserve_original_order(self):
        event=upstream(1)['events'][0]
        event.update(sources=[{'text':'随后倒地。','start':8}],contexts=[{'text':'雷先喝水，','start':0}])
        query=event_query(event)
        self.assertIn('雷先喝水，随后倒地。',query)

    def test_missing_index_failure_keeps_full_upstream_report(self):
        result=retrieve_events(filtered(1),index_directory='missing_retrieval_index_for_test')
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['filtering']['stage'],'filtering')
        self.assertIn('索引',result['items'][0]['error'])
