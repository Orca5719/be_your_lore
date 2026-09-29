import unittest
from agent_pipeline.filtering import filter_events
from agent_pipeline.retrieval import retrieve_events
from test_agent_pipeline_filtering import upstream,FakeLLM
from test_agent_pipeline_retrieval import FakeRetriever


class ProcessDetailTests(unittest.TestCase):
    def test_process_detail_not_independently_judged_but_required_context_survives(self):
        value=upstream(5)
        for event,text in zip(value['events'],['手臂液化','手臂进入主机','屏幕变化','文件切换','显示震中图像']):event['event']=text
        value['events'][4]['context_ids']=['S3']
        rows=[dict(event_id='E'+str(i),decision='ignore' if i in (3,4) else 'keep',reason_code='process_detail' if i in (3,4) else 'mechanism') for i in range(1,6)]
        result=filter_events(value,llm=FakeLLM([{'decisions':rows}]))
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['decisions'][2]['support_only'])
        self.assertEqual([e['id'] for e in result['ignored_events']],['E4'])
        retriever=FakeRetriever();retrieve_events(result,retriever=retriever)
        self.assertEqual(len(retriever.queries),3)

    def test_explicit_special_screen_mechanism_is_not_dropped_by_keywords(self):
        value=upstream(1);value['events'][0]['event']='主机断电后屏幕仍显示已删除的机密档案'
        result=filter_events(value,llm=FakeLLM([{'decisions':[dict(event_id='E1',decision='keep',reason_code='mechanism')]}]))
        self.assertEqual(result['selected_events'][0]['id'],'E1')
