import json
import unittest
from agent_pipeline.understanding import understand,split_spans
from agent_pipeline.llm import call_json


def row(**changes):
    base=dict(actors=['雷'],event='喝水',mental_state=None,explicit=True,modality='observed',conditions=[],source_ids=['S1'],context_ids=[])
    base.update(changes);return base


class Fake:
    device='cpu';last_generation={}
    def __init__(self,replies):self.replies=iter(replies);self.messages=[]
    def _generate(self,messages,max_new_tokens):
        self.messages.append(messages)
        return next(self.replies,'{"events":[]}')


class CompletionTests(unittest.TestCase):
    def test_unique_prior_name_is_attached_without_changing_proposition(self):
        raw=row(event='感到熟悉',mental_state='熟悉',source_ids=['S2'])
        result=understand('雷喝水。他感到熟悉。',llm=Fake([json.dumps({'events':[row(),raw]})]))
        self.assertEqual(len(result['events']),2)
        self.assertEqual(result['events'][1]['context_ids'],['S1'])
        self.assertEqual(result['events'][1]['event'],raw['event'])
        self.assertEqual(result['events'][1]['reference_repairs'],['unique_prior_name_anchor_added'])

    def test_coverage_gap_gets_one_targeted_recovery(self):
        fake=Fake(['{"events":[]}',json.dumps({'events':[row()]})])
        result=understand('雷喝水。',llm=fake)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['calls']),2)
        self.assertEqual(result['calls'][1]['purpose'],'coverage_recovery')

    def test_schema_invalid_json_is_retried_and_preserved(self):
        validate=lambda v: None if set(v)=={'events'} and isinstance(v['events'],list) else (_ for _ in ()).throw(ValueError('bad schema'))
        value,call=call_json(Fake(['{"other":[]}','{"events":[]}']),[],validator=validate)
        self.assertEqual(value,{'events':[]})
        self.assertEqual(len(call['attempts']),2)
        self.assertEqual(call['attempts'][0]['status'],'error')

    def test_long_unpunctuated_span_is_split_without_losing_text(self):
        text='雷'*600
        spans=split_spans(text)
        self.assertGreater(len(spans),1)
        self.assertEqual(''.join(v['text'] for v in spans.values()),text)
        self.assertTrue(all(text[v['start']:v['end']]==v['text'] for v in spans.values()))

    def test_identical_event_is_deduplicated_but_different_conditions_preserved(self):
        first=row();second=row(conditions=['受伤时'])
        result=understand('雷受伤时喝水。',llm=Fake([json.dumps({'events':[first,first,second]})]))
        self.assertEqual(len(result['events']),2)
        self.assertEqual(result['duplicate_candidates'],1)

    def test_events_remain_in_source_order_after_context_enrichment(self):
        values=[row(source_ids=['S2']),row(source_ids=['S1']),row(source_ids=['S3'])]
        result=understand('雷喝水。他起身。雷出门。',llm=Fake([json.dumps({'events':values})]))
        self.assertEqual([e['source_ids'] for e in result['events']],[['S1'],['S2'],['S3']])

    def test_repeated_action_at_distinct_offsets_is_not_deduplicated(self):
        values=[row(source_ids=['S1']),row(source_ids=['S2'])]
        result=understand('雷喝水。雷喝水。',llm=Fake([json.dumps({'events':values})]))
        self.assertEqual(len(result['events']),2)

    def test_missing_optional_fields_have_logged_defaults(self):
        raw=row();del raw['mental_state'];del raw['conditions'];del raw['context_ids']
        result=understand('雷喝水。',llm=Fake([json.dumps({'events':[raw]})]))
        self.assertEqual(result['status'],'ok')
        self.assertIsNone(result['events'][0]['mental_state'])
        self.assertEqual(set(result['events'][0]['defaulted_fields']),{'mental_state','conditions','context_ids'})

    def test_runtime_error_is_not_repeated_in_coverage_recovery(self):
        class Broken(Fake):
            def _generate(self,messages,max_new_tokens):
                self.messages.append(messages)
                raise RuntimeError('model runtime failure')
        fake=Broken([])
        result=understand('雷喝水。',llm=fake)
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(fake.messages),1)

    def test_failed_window_is_not_hidden_by_later_context_reference(self):
        class BrokenFirst(Fake):
            def _generate(self,messages,max_new_tokens):
                self.messages.append(messages)
                if len(self.messages)==1:raise RuntimeError('first window failed')
                return json.dumps({'events':[row(source_ids=['S2'],context_ids=['S1'])]})
        result=understand('甲'*179+'。雷喝水。',llm=BrokenFirst([]))
        self.assertEqual(result['status'],'partial')
        self.assertFalse(result['processing_complete'])
        self.assertEqual(result['unrecovered_failed_source_ids'],['S1'])

    def test_scalar_condition_and_field_case_normalization_preserve_content(self):
        raw=row();raw['conditions']='受伤时';raw['source_Ids']=raw.pop('source_ids')
        result=understand('雷受伤时喝水。',llm=Fake([json.dumps({'events':[raw]})]))
        self.assertEqual(result['status'],'ok')
        event=result['events'][0]
        self.assertEqual(event['conditions'],['受伤时'])
        self.assertIn('conditions_scalar_wrapped',event['wire_normalizations'])
        self.assertIn('field_case_normalized:source_Ids',event['wire_normalizations'])

    def test_unsupported_enum_is_targeted_for_recovery_even_if_context_covered(self):
        bad=row(modality='mental_state')
        good=row(event='起身',source_ids=['S2'],context_ids=['S1'])
        fake=Fake([json.dumps({'events':[bad,good]}),json.dumps({'events':[row()]})])
        result=understand('雷喝水。雷起身。',llm=fake)
        self.assertEqual(len(result['events']),2)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['rejected'][0]['recovered_event_ids'],['E1'])

    def test_optional_defaults_are_also_applied_after_llm_reference_repair(self):
        raw=row();del raw['mental_state'];del raw['conditions'];del raw['context_ids']
        fake=Fake([json.dumps({'events':[raw]}),'{"repairs":[{"candidate_index":1,"context_ids":["S2"]}]}'])
        result=understand('他喝水。这个人名叫雷。',llm=fake)
        self.assertEqual(result['events'][0]['context_ids'],['S2'])
        self.assertIsNone(result['events'][0]['mental_state'])
        self.assertTrue(result['events'][0]['review_required'])
