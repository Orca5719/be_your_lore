import json
import unittest
from agent_pipeline.understanding import understand
from test_agent_pipeline_understanding import Fake,event


class RecoveryMappingTests(unittest.TestCase):
    def test_recovery_only_receives_rejections_for_its_target_window(self):
        text='甲'*100+'。'+'乙'*100+'。'
        bad1=dict(event(),actors=['系统'],event='甲发生变化',source_ids=['S1'])
        bad2=dict(event(),actors=['装置'],event='乙发生变化',source_ids=['S2'])
        model=Fake([
            json.dumps({'events':[bad1]}),json.dumps({'events':[bad2]}),
            json.dumps({'events':[]}),json.dumps({'events':[]}),
        ])
        understand(text,llm=model)
        recovery_payloads=[json.loads(messages[1]['content']) for messages in model.messages if 'rejected_candidates' in json.loads(messages[1]['content'])]
        self.assertEqual(len(recovery_payloads),2)
        self.assertEqual([[c['candidate_id'] for c in payload['rejected_candidates']] for payload in recovery_payloads],[['W1C1'],['W2C1']])

    def test_invalid_reference_can_be_removed_without_losing_valid_source(self):
        bad=dict(event(),source_ids=['S1','S99'])
        good=dict(event(),replaces=['W1C1'])
        result=understand('雷喝水。',llm=Fake([json.dumps({'events':[bad]}),json.dumps({'events':[good]})]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['events'][0]['source_ids'],['S1'])

    def test_explicit_corrected_actor_replaces_rejection_but_retains_raw(self):
        bad=dict(event(),actors=['系统'],event='屏幕发生变化')
        good=dict(event(),actors=['屏幕'],event='屏幕发生变化',replaces=['W1C1'])
        result=understand('屏幕发生变化。',llm=Fake([json.dumps({'events':[bad]}),json.dumps({'events':[good]})]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['rejected'][0]['raw_event']['actors'],['系统'])
        self.assertEqual(result['rejected'][0]['recovered_event_ids'],['E1'])
        self.assertIn('recovered_candidate_reinterpreted',result['events'][0]['review_reasons'])

    def test_wrong_source_cannot_resolve_rejected_candidate(self):
        bad=dict(event(),actors=['系统'],event='屏幕发生变化')
        first=dict(event(),actors=['屏幕'],event='屏幕发生变化')
        wrong=dict(event(),actors=['雷'],event='雷喝水',source_ids=['S2'],replaces=['W1C1'])
        result=understand('屏幕发生变化。雷喝水。',llm=Fake([json.dumps({'events':[bad,dict(wrong,replaces=[])]}),json.dumps({'events':[wrong,first]})]))
        self.assertEqual(result['status'],'partial')
        self.assertTrue(result['failure_reasons']['unresolved_candidates'])

    def test_covering_source_without_explicit_mapping_does_not_hide_reject(self):
        bad=dict(event(),actors=['系统'],event='屏幕发生变化')
        good=dict(event(),actors=['屏幕'],event='屏幕发生变化')
        result=understand('屏幕发生变化。',llm=Fake([json.dumps({'events':[bad]}),json.dumps({'events':[good]})]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['failure_reasons']['unresolved_candidates'][0]['candidate_id'],'W1C1')
