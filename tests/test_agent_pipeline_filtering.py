import copy
import json
import unittest
from agent_pipeline.filtering import filter_events


class FakeLLM:
    device='cpu'
    load_seconds=0
    def __init__(self, replies):self.replies=iter(replies);self.messages=[]
    def _generate(self,messages,max_new_tokens):
        self.messages.append(messages)
        reply=next(self.replies,RuntimeError('fake model has no queued reply'))
        if isinstance(reply,Exception):raise reply
        return json.dumps(reply,ensure_ascii=False) if isinstance(reply,dict) else reply


def upstream(count=2,status='ok'):
    return dict(stage='understanding',status=status,semantic_verification='not_verified',events=[dict(id=f'E{i}',actors=['雷'],event='喝水' if i==1 else '左心脏寄宿亚巴顿',mental_state=None,explicit=True,modality='observed',conditions=[],source_ids=[f'S{i}'],context_ids=[],sources=[{'text':'原文'}]) for i in range(1,count+1)])


def decision(eid,choice='keep'):
    return dict(event_id=eid,decision=choice,reason_code='routine' if choice=='ignore' else 'unclear' if choice=='review' else 'mechanism')


class FilteringTests(unittest.TestCase):
    def test_decisions_keep_original_events_and_never_mutate_input(self):
        report=upstream();original=copy.deepcopy(report)
        result=filter_events(report,llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2')]}]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual([e['id'] for e in result['selected_events']],['E2'])
        self.assertEqual(result['events'],original['events'])
        self.assertEqual(report,original)

    def test_missing_decision_is_review_not_ignore(self):
        result=filter_events(upstream(),llm=FakeLLM([{'decisions':[decision('E1','ignore')]}]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['decisions'][1]['decision'],'review')
        self.assertEqual(result['pending_events'][0]['id'],'E2')

    def test_conflicting_duplicate_does_not_get_silently_accepted(self):
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[decision('E1'),decision('E1','ignore')]}]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['decisions'][0]['decision'],'review')

    def test_unknown_id_preserves_rejection_and_partial(self):
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[decision('E1'),decision('E99')]}]))
        self.assertEqual(result['status'],'partial')
        self.assertTrue(result['rejected'])

    def test_failed_window_is_review_and_does_not_repeat_runtime_failure(self):
        model=FakeLLM([RuntimeError('GPU failure')])
        result=filter_events(upstream(),llm=model)
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(result['pending_events']),2)
        self.assertEqual(len(model.messages),1)

    def test_partial_upstream_is_not_masked_by_successful_filter(self):
        result=filter_events(upstream(1,'partial'),llm=FakeLLM([{'decisions':[decision('E1')]}]))
        self.assertEqual(result['status'],'partial')
        self.assertTrue(result['filtering_complete'])
        self.assertFalse(result['processing_complete'])

    def test_empty_events_do_not_load_model(self):
        result=filter_events(upstream(0))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['calls'],[])
        self.assertFalse(result['model_loaded_this_request'])

    def test_invalid_ids_and_input_rejected_before_model(self):
        report=upstream();report['events'][1]['id']='E1'
        with self.assertRaises(ValueError):filter_events(report)
        with self.assertRaises(ValueError):filter_events({'events':[]})

    def test_batched_calls_only_include_bounded_neighbor_summaries(self):
        report=upstream(25);report['text']='不应在每个窗口重复的完整故事'
        model=FakeLLM([{'decisions':[decision(f'E{i}') for i in range(1,9)]},{'decisions':[decision(f'E{i}') for i in range(9,17)]},{'decisions':[decision(f'E{i}') for i in range(17,25)]},{'decisions':[decision('E25')]}])
        result=filter_events(report,llm=model)
        self.assertEqual(len(result['selected_events']),25)
        payload=json.loads(model.messages[1][1]['content'])
        self.assertNotIn('story_text',payload)
        self.assertLessEqual(len(payload['context_events']),4)
        self.assertEqual([e['id'] for e in payload['context_events']],['E7','E8','E17','E18'])
        self.assertTrue(all(set(e)<=set(('id','actors','event','mental_state','explicit','modality','conditions')) for e in payload['context_events']))

    def test_json_retry_retains_raw_outputs(self):
        result=filter_events(upstream(1),llm=FakeLLM(['broken',{'decisions':[decision('E1')]}]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['calls'][0]['attempts']),2)

    def test_unresolved_actor_does_not_block_a_routine_ignore(self):
        report=upstream(1);report['events'][0].update(review_required=True,review_reasons=['unresolved_actor'])
        result=filter_events(report,llm=FakeLLM([{'decisions':[decision('E1','ignore')]}]))
        self.assertEqual(result['decisions'][0]['decision'],'ignore')

    def test_unresolved_actor_cannot_be_sent_to_judge_as_keep(self):
        report=upstream(1);report['events'][0].update(review_required=True,review_reasons=['unresolved_actor'])
        result=filter_events(report,llm=FakeLLM([{'decisions':[decision('E1')]}]))
        self.assertEqual(result['decisions'][0]['decision'],'review')
        self.assertEqual(result['decisions'][0]['model_decision'],'keep')

    def test_only_invalid_target_decision_is_retried(self):
        bad={'decisions':[decision('E1'),dict(event_id='E2',decision='keep',reason_code='routine')]}
        repair={'decisions':[decision('E2','ignore')]}
        model=FakeLLM([bad,repair])
        result=filter_events(upstream(2),llm=model)
        self.assertEqual(result['status'],'ok')
        self.assertEqual([d['decision'] for d in result['decisions']],['keep','ignore'])
        self.assertEqual(len(model.messages),2)
        payload=json.loads(model.messages[1][1]['content'])
        self.assertEqual([e['id'] for e in payload['target_events']],['E2'])
        self.assertEqual(result['rejected'][0]['recovered_event_ids'],['E2'])

    def test_reason_code_copied_into_decision_is_safely_normalized(self):
        copied={'decisions':[dict(event_id='E1',decision='routine',reason_code='routine')]}
        result=filter_events(upstream(1),llm=FakeLLM([copied]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['decisions'][0]['decision'],'ignore')
        self.assertEqual(result['decisions'][0]['wire_normalizations'],['decision_from_reason_code'])

    def test_conflicting_decision_and_reason_code_is_not_normalized(self):
        bad={'decisions':[dict(event_id='E1',decision='routine',reason_code='mechanism')]}
        result=filter_events(upstream(1),llm=FakeLLM([bad,RuntimeError('repair failed')]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['decisions'][0]['decision'],'review')

    def test_decision_for_known_context_event_is_audited_not_failure(self):
        first={'decisions':[decision(f'E{i}') for i in range(1,9)]+[decision('E9','ignore')]}
        model=FakeLLM([first,{'decisions':[decision('E9')]}])
        result=filter_events(upstream(9),llm=model)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['extraneous_decisions'][0]['raw_decision']['event_id'],'E9')

    def test_extra_fact_rewrite_is_rejected(self):
        row=decision('E1');row['event']='掌控时间'
        result=filter_events(upstream(1),llm=FakeLLM([{'decisions':[row]}]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['pending_events'][0]['event'],'喝水')

    def test_malformed_review_metadata_is_rejected_before_generation(self):
        report=upstream(1);report['events'][0]['review_reasons']=[{}]
        model=FakeLLM([{'decisions':[decision('E1','ignore')]}])
        with self.assertRaises(ValueError):filter_events(report,llm=model)
        self.assertEqual(model.messages,[])

    def test_upstream_error_remains_error_even_with_empty_events(self):
        result=filter_events(upstream(0,'error'))
        self.assertEqual(result['status'],'error')
        self.assertFalse(result['processing_complete'])
