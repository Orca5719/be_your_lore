import json
import unittest
from agent_pipeline.filtering import filter_events
from test_agent_pipeline_filtering import FakeLLM,upstream,decision
from test_agent_pipeline_understanding import Fake,event
from agent_pipeline.understanding import understand


class ReportedFailureTests(unittest.TestCase):
    def test_filter_model_receives_original_provenance(self):
        report=upstream(1);report['text']='测试原文'
        model=FakeLLM([{'decisions':[decision('E1')]}])
        filter_events(report,llm=model)
        payload=json.loads(model.messages[0][1]['content'])
        self.assertNotIn('story_text',payload)
        self.assertEqual(payload['target_events'][0]['sources'],report['events'][0]['sources'])

    def test_required_context_is_not_silently_ignored(self):
        report=upstream(2)
        report['events'][0].update(source_ids=['S1'])
        report['events'][1].update(context_ids=['S1'],conditions=['喝水后'])
        result=filter_events(report,llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2')]}]))
        self.assertEqual(result['decisions'][0]['decision'],'keep')
        self.assertEqual(result['decisions'][0]['model_decision'],'ignore')
        self.assertEqual(result['decisions'][0]['required_by_event_ids'],['E2'])
        self.assertTrue(result['decisions'][0]['support_only'])

    def test_understanding_receives_connected_sentence(self):
        model=Fake([json.dumps({'events':[event()]},ensure_ascii=False)])
        understand('雷喝了一口水。',llm=model)
        payload=json.loads(model.messages[0][1]['content'])
        self.assertNotIn('story_text',payload)
        self.assertEqual(payload['target_spans'],{'S1':'雷喝了一口水。'})

    def test_partial_exposes_exact_failure_reasons(self):
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[]}]))
        self.assertEqual(result['failure_reasons']['pending_event_ids'],['E1'])
        self.assertFalse(result['processing_complete'])

    def test_invalid_source_ids_rejected_before_inference(self):
        report=upstream(1);report['events'][0]['source_ids']=[{}]
        model=FakeLLM([{'decisions':[decision('E1','ignore')]}])
        with self.assertRaises(ValueError):filter_events(report,llm=model)
        self.assertEqual(model.messages,[])

    def test_null_context_ids_rejected_before_inference(self):
        report=upstream(1);report['events'][0].update(context_ids=None,conditions=['某条件'])
        model=FakeLLM([{'decisions':[decision('E1')]}])
        with self.assertRaises(ValueError):filter_events(report,llm=model)
        self.assertEqual(model.messages,[])

    def test_reason_codes_render_without_generated_claims(self):
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[dict(event_id='E1',decision='ignore',reason_code='routine')]}]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['decisions'][0]['reason_code'],'routine')
        self.assertEqual(result['decisions'][0]['reason'],'普通日常或背景，无特殊影响')

    def test_unknown_reason_code_and_contradictory_decision_rejected(self):
        for code in ['invented','routine']:
            result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[dict(event_id='E1',decision='keep',reason_code=code)]}]))
            self.assertEqual(result['status'],'partial')

    def test_free_form_model_reason_is_not_accepted_as_claim(self):
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[dict(event_id='E1',decision='keep',reason='喝水导致死亡')]}]))
        self.assertEqual(result['status'],'partial')
        self.assertNotIn('死亡',result['decisions'][0]['reason'])

    def test_named_prior_context_alone_does_not_block_routine_ignore(self):
        report=upstream(1);report['events'][0]['review_reasons']=['actor_resolution_proposed_by_model']
        result=filter_events(report,llm=FakeLLM([{'decisions':[dict(event_id='E1',decision='ignore',reason_code='routine')]}]))
        self.assertEqual(result['status'],'ok')

    def test_dependency_support_is_preserved_transitively(self):
        report=upstream(3)
        report['events'][1]['context_ids']=['S1']
        report['events'][2]['context_ids']=['S2']
        result=filter_events(report,llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2','ignore'),decision('E3')]}]))
        self.assertEqual([e['id'] for e in result['selected_events']],['E1','E2','E3'])
        self.assertEqual(result['decisions'][0]['required_by_event_ids'],['E2'])
