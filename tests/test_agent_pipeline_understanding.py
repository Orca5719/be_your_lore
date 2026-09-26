import json
import unittest
from agent_pipeline.understanding import understand, split_spans, windows


class Fake:
    device='cpu'
    last_generation={'seconds':1}
    def __init__(self,replies):self.replies=iter(replies);self.messages=[]
    def _generate(self,messages,max_new_tokens):
        self.messages.append(messages)
        return next(self.replies,'{"repairs":[]}')


def event(**changes):
    return dict(actors=['雷'],event='喝了一口水',mental_state=None,explicit=True,
                modality='observed',conditions=[],source_ids=['S1'],context_ids=[],**changes)


class UnderstandingTests(unittest.TestCase):
    def test_daily_actions_are_retained_and_sources_located(self):
        result=understand('雷喝了一口水。',llm=Fake([json.dumps({'events':[event()]})]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['events'][0]['event'],'喝了一口水')
        self.assertEqual(result['events'][0]['sources'][0]['text'],'雷喝了一口水。')
        self.assertEqual(result['coverage']['missing_source_ids'],[])

    def test_empty_output_exposes_coverage_gap(self):
        result=understand('雷拥有双心脏。',llm=Fake(['{"events":[]}']))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['coverage']['missing_source_ids'],['S1'])

    def test_explicit_non_event_span_is_audited_without_forcing_fake_event(self):
        reply={'events':[],'non_event_span_ids':['S1']}
        result=understand('随后，',llm=Fake([json.dumps(reply,ensure_ascii=False)]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['events'],[])
        self.assertEqual(result['coverage']['non_event_source_ids'],['S1'])
        self.assertEqual(result['coverage']['missing_source_ids'],[])

    def test_recovery_context_only_candidate_is_audited_without_partial(self):
        first_row=event();first_row.update(event='雷折起纸巾',source_ids=['S1'])
        recovered_target=event();recovered_target.update(event='纸巾被放在桌边',source_ids=['S2'])
        repeated_context=event();repeated_context.update(event='雷折起纸巾',source_ids=['S1'])
        first={'events':[first_row]}
        recovered={'events':[
            recovered_target,
            repeated_context,
        ],'non_event_span_ids':[]}
        result=understand('雷折起纸巾。放在桌边。',llm=Fake([json.dumps(first,ensure_ascii=False),json.dumps(recovered,ensure_ascii=False)]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['events']),2)
        self.assertEqual(result['extraneous_candidates'][0]['reason'],'coverage_recovery_context_only_output')

    def test_bad_reference_rejected_without_losing_valid_event(self):
        bad=dict(event(),source_ids=['S99'])
        result=understand('雷喝了一口水。',llm=Fake([json.dumps({'events':[event(),bad]})]))
        self.assertEqual(len(result['events']),1)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(len(result['rejected']),1)

    def test_inference_is_marked_not_converted_to_observed(self):
        inferred=dict(event(),event='可能紧张',explicit=False,modality='inferred')
        result=understand('雷反复攥紧手指。',llm=Fake([json.dumps({'events':[inferred]})]))
        self.assertFalse(result['events'][0]['explicit'])
        invalid=dict(inferred,modality='observed')
        result=understand('雷反复攥紧手指。',llm=Fake([json.dumps({'events':[invalid]})]))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['events'],[])

    def test_format_retry_preserves_both_attempts(self):
        result=understand('雷喝了一口水。',llm=Fake(['bad',json.dumps({'events':[event()]})]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(result['calls'][0]['attempts']),2)
        self.assertEqual(result['calls'][0]['attempts'][0]['raw_output'],'bad')

    def test_failed_window_does_not_fake_normal_empty(self):
        result=understand('雷喝了一口水。',llm=Fake(['bad','bad']))
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['coverage']['missing_source_ids'],['S1'])

    def test_spans_exact_and_windows_cover_without_duplicate_target(self):
        text='雷喝水，走到窗边。'*30
        spans=split_spans(text)
        batches=windows(spans)
        target_ids=[key for batch in batches for key in batch['target_ids']]
        self.assertEqual(target_ids,list(spans))
        self.assertTrue(all(text[v['start']:v['end']]==v['text'] for v in spans.values()))
        self.assertGreater(len(batches),1)

    def test_windows_only_include_two_nearest_prior_spans(self):
        spans={f'S{i}':dict(text='甲'*100,start=i*100,end=(i+1)*100) for i in range(1,7)}
        batches=windows(spans)
        self.assertEqual([b['target_ids'] for b in batches],[['S1'],['S2'],['S3'],['S4'],['S5'],['S6']])
        self.assertEqual(batches[0]['context_ids'],[])
        self.assertEqual(batches[3]['context_ids'],['S2','S3'])
        self.assertTrue(all(len(b['context_ids'])<=2 for b in batches))

    def test_empty_and_long_input_rejected_before_model_loading(self):
        for text in [' ','雷'*801]:
            with self.assertRaises(ValueError):understand(text)

    def test_resolved_actor_needs_named_antecedent_in_references(self):
        from agent_pipeline.understanding import validate_event
        row=dict(event(),event='感到模糊的熟悉',mental_state='模糊的熟悉',source_ids=['S2'])
        with self.assertRaises(ValueError):validate_event(row,split_spans('雷停下脚步。他感到模糊的熟悉。'),{'target_ids':['S1','S2'],'context_ids':[]})
        grounded=dict(row,context_ids=['S1'])
        result=understand('雷停下脚步。他感到模糊的熟悉。',llm=Fake([json.dumps({'events':[grounded]})]))
        self.assertEqual(result['events'][0]['contexts'][0]['text'],'雷停下脚步。')
        self.assertTrue(result['events'][0]['explicit'])

    def test_actor_not_in_story_is_rejected(self):
        row=dict(event(),actors=['银河皇帝'])
        result=understand('雷喝了一口水。',llm=Fake([json.dumps({'events':[row]})]))
        self.assertEqual(result['events'],[])
        self.assertEqual(len(result['rejected']),1)

    def test_unresolved_pronoun_is_preserved_not_guessed(self):
        row=dict(event(),actors=['他'],event='喝水',source_ids=['S2'])
        result=understand('雷与德尔塔站在窗前。他喝水。',llm=Fake([json.dumps({'events':[row]})]))
        self.assertEqual(result['events'][0]['actors'],['他'])

    def test_only_context_cannot_generate_new_event(self):
        from agent_pipeline.understanding import validate_event
        spans=split_spans('雷喝水。他起身。')
        batch={'target_ids':['S2'],'context_ids':['S1']}
        with self.assertRaises(ValueError):validate_event(event(),spans,batch)

    def test_reference_repair_only_adds_context_and_preserves_semantics(self):
        from agent_pipeline.understanding import repair_actor_context,validate_event
        row=dict(event(),event='感到模糊的熟悉',mental_state='模糊的熟悉',source_ids=['S3'])
        spans=split_spans('雷停下脚步。德尔塔望向雷。他感到模糊的熟悉。')
        batch={'target_ids':list(spans),'context_ids':[]}
        fixes,call=repair_actor_context(Fake(['{"repairs":[{"candidate_index":1,"context_ids":["S1"]}]}']),[{'candidate_index':1,'event':row}],spans,batch)
        repaired=validate_event(dict(row,context_ids=fixes[1]),spans,batch)
        self.assertEqual(repaired['context_ids'],['S1'])
        self.assertEqual(repaired['event'],row['event'])
        self.assertEqual(call['status'],'ok')

    def test_repair_cannot_change_actor_or_event(self):
        from agent_pipeline.understanding import repair_actor_context
        row=dict(event(),source_ids=['S3'])
        spans=split_spans('雷停下脚步。德尔塔望向雷。他喝水。')
        fixes,call=repair_actor_context(Fake(['{"repairs":[{"candidate_index":1,"context_ids":["S1"],"event":"逃走"}]}']),[{'candidate_index':1,'event':row}],spans,{'target_ids':list(spans),'context_ids':[]})
        self.assertEqual(fixes,{})
        self.assertEqual(call['status'],'error')
        self.assertEqual(row['event'],'喝了一口水')

    def test_reference_hints_show_all_mentions_without_auto_selection(self):
        from agent_pipeline.understanding import repair_actor_context
        row=dict(event(),source_ids=['S3'])
        fake=Fake(['{"repairs":[]}']);spans=split_spans('雷停下脚步。德尔塔望向雷。他喝水。')
        fixes,_=repair_actor_context(fake,[{'candidate_index':1,'event':row}],spans,{'target_ids':list(spans),'context_ids':[]})
        payload=json.loads(fake.messages[0][1]['content'])
        self.assertEqual(payload['candidates'][0]['actor_mentions'],{'雷':['S1','S2']})
        self.assertEqual(fixes,{})

    def test_each_model_payload_is_local_and_omits_full_story(self):
        text='甲'*100+'。'+'乙'*100+'。'+'丙'*100+'。'
        replies=[json.dumps({'events':[]}) for _ in range(6)]
        model=Fake(replies)
        understand(text,llm=model)
        payloads=[json.loads(messages[1]['content']) for messages in model.messages]
        self.assertTrue(payloads)
        self.assertTrue(all('story_text' not in payload for payload in payloads))
        self.assertTrue(all(len(payload['context_spans'])<=2 for payload in payloads))
