import unittest
from story_extraction import story_spans, decode_facts, extract_story

class StoryProtocolTests(unittest.TestCase):
    def test_source_positions_and_multiple_span_context(self):
        text='第三话之后，雷的左胸疼痛。'
        spans=story_spans(text)
        self.assertTrue(all(text[s['start']:s['end']]==s['text'] for s in spans.values()))
        value={'facts':[{'subject':'雷','predicate':'感到','object':'左胸疼痛',
                         'type':'character_attribute','source_ids':['S2'],'context_ids':['S1']}]}
        result=decode_facts(value,text,spans)
        self.assertEqual(result['facts'][0]['source_text'],'雷的左胸疼痛。')
        self.assertEqual(result['facts'][0]['context_text'],'第三话之后，')
        self.assertEqual(result['facts'][0]['id'],'F1')

    def test_empty_facts_are_valid(self):
        result=decode_facts({'facts':[]},'雷喝水。',story_spans('雷喝水。'))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['facts'],[])

    def test_one_bad_fact_does_not_drop_good_fact(self):
        text='雷有两颗心脏。'
        good={'subject':'雷','predicate':'拥有','object':'两颗心脏','type':'character_attribute','source_ids':['S1']}
        value={'facts':[good,dict(good,source_ids=['S99']),dict(good,subject='银河皇帝'),good]}
        result=decode_facts(value,text,story_spans(text))
        self.assertEqual(result['status'],'partial')
        self.assertEqual(len(result['facts']),1)
        self.assertEqual(len(result['rejected']),2)
        self.assertEqual(result['duplicates'],[4])
        self.assertEqual(result['rejected'][0]['fact_index'],2)

    def test_invalid_types_unknown_fields_and_empty_source(self):
        good={'subject':'雷','predicate':'不知道','object':'秘密','type':'character_knowledge','source_ids':['S1']}
        for change in [{'type':'fake'},{'source_ids':[]},{'predicate':''},{'text':'new'}]:
            result=decode_facts({'facts':[dict(good,**change)]},'雷不知道秘密。',story_spans('雷不知道秘密。'))
            self.assertEqual(result['status'],'error')
        with self.assertRaises(ValueError):decode_facts({'facts':'bad'},'雷不知道秘密。',{})
        with self.assertRaises(ValueError):story_spans('  ')

    def test_adapter_uses_new_prompt_and_keeps_raw_output(self):
        class Fake:
            device='cpu'
            last_generation={'seconds':1}
            def _generate(self,messages,**kwargs):
                self.messages=messages
                return '{"facts":[]}'
        fake=Fake()
        result=extract_story('雷吃饭。',judge=fake)
        self.assertEqual(result['facts'],[])
        self.assertIn('source_spans',fake.messages[1]['content'])
        self.assertEqual(result['raw_output'],'{"facts":[]}')
        self.assertEqual(result['prompt_version'],'story-extraction-v2')

    def test_importance_review_must_account_for_every_candidate(self):
        from story_extraction import apply_importance_review
        text='雷喝水。他有两颗心脏。'
        rows=[{'subject':'雷','predicate':'喝','object':'水','type':'object_state','source_ids':['S1']},
              {'subject':'雷','predicate':'拥有','object':'两颗心脏','type':'character_attribute','source_ids':['S2'],'context_ids':['S1']}]
        result=decode_facts({'facts':rows},text,story_spans(text))
        reviewed=apply_importance_review(result,{'keep':['F2'],'discard':[{'id':'F1','reason':'普通喝水'}]},story_spans(text))
        self.assertEqual([f['id'] for f in reviewed['facts']],['F2'])
        self.assertEqual(reviewed['discarded'][0]['reason'],'普通喝水')
        for decision in [{'keep':[],'discard':[]}, {'keep':['F1','F2','F3'],'discard':[]},
                         {'keep':['F1','F2'],'discard':[{'id':'F1','reason':'重复'}]}]:
            with self.assertRaises(ValueError):apply_importance_review(result,decision,story_spans(text))

    def test_bad_importance_review_preserves_candidate_and_validation_error(self):
        import json
        class Fake:
            device='cpu'
            last_generation={'seconds':1}
            def __init__(self):self.calls=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    row={'subject':'雷','predicate':'拥有','object':'两颗心脏','type':'character_attribute','source_ids':['S1']}
                    return json.dumps({'facts':[row,dict(row,source_ids=['S99'])]})
                return '{"keep":[],"discard":[]}'
        result=extract_story('雷有两颗心脏。',judge=Fake())
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['facts'],[])
        self.assertEqual(len(result['pending_facts']),1)
        self.assertEqual(len(result['candidates']),1)
        self.assertEqual(len(result['rejected']),1)

    def test_unary_state_accepts_null_missing_and_empty_object(self):
        text='月城的银色手臂如液体一般化开，渗透进了实验室的主机中。'
        state={'subject':'月城的银色手臂','predicate':'如液体一般化开','type':'object_state','source_ids':['S1']}
        for row in [state,dict(state,object=None),dict(state,object='')]:
            result=decode_facts({'facts':[row]},text,story_spans(text))
            self.assertEqual(result['status'],'ok')
            self.assertIsNone(result['facts'][0]['object'])
        invalid=dict(state,object={'guess':'液体'})
        self.assertEqual(decode_facts({'facts':[invalid]},text,story_spans(text))['status'],'error')

    def test_omitted_subject_adds_unique_same_sentence_antecedent(self):
        text='月城的银色手臂如液体一般化开，渗透进了实验室的主机中。'
        row={'subject':'月城的银色手臂','predicate':'渗透进','object':'实验室的主机中','type':'object_state','source_ids':['S2']}
        result=decode_facts({'facts':[row]},text,story_spans(text))
        self.assertEqual(result['facts'][0]['context_ids'],['S1'])
        self.assertIn('月城的银色手臂',result['facts'][0]['context_text'])
        text='月城的银色手臂停下。屏幕显示图像。'
        row['source_ids']=['S2']
        self.assertEqual(decode_facts({'facts':[row]},text,story_spans(text))['status'],'error')

    def test_ids_stay_aligned_with_original_candidates(self):
        text='月城的银色手臂化开。'
        row={'subject':'月城的银色手臂','predicate':'化开','object':None,'type':'object_state','source_ids':['S1']}
        result=decode_facts({'facts':[dict(row,type='fake'),row]},text,story_spans(text))
        self.assertEqual(result['facts'][0]['id'],'F2')
        self.assertEqual(result['rejected'][0]['id'],'F1')

    def test_conflicting_review_isolated_to_pending_fact(self):
        from story_extraction import apply_importance_review
        text='雷失明。雷的长剑断裂。'
        rows=[{'subject':'雷','predicate':'失明','object':None,'type':'character_attribute','source_ids':['S1']},
              {'subject':'雷的长剑','predicate':'断裂','object':None,'type':'object_state','source_ids':['S2']}]
        result=decode_facts({'facts':rows},text,story_spans(text))
        value={'keep':['F1','F2'],'discard':[{'id':'F2','reason':'错误判断'}]}
        result=apply_importance_review(result,value,story_spans(text),recover=True)
        self.assertEqual(result['status'],'partial')
        self.assertEqual([f['id'] for f in result['facts']],['F1'])
        self.assertEqual([f['id'] for f in result['pending_facts']],['F2'])
        self.assertEqual(result['review_issues'][0]['id'],'F2')


    def test_discarded_action_does_not_fail_subject_grounding(self):
        import json
        class Fake:
            device='cpu'
            last_generation={'seconds':1}
            def __init__(self):self.calls=[]
            def _generate(self,messages,**kwargs):
                self.calls.append(messages)
                if len(self.calls)==1:
                    return json.dumps({'facts':[
                        {'subject':'米娅','predicate':'幻化出','object':'长剑','type':'character_attribute','source_ids':['S1']},
                        {'subject':'米娅','predicate':'举起','object':'长剑','type':'object_state','source_ids':['S2']}]})
                return json.dumps({'decision':'keep' if len(self.calls)==2 else 'discard','reason':'特殊能力' if len(self.calls)==2 else '普通战斗动作'})
        fake=Fake()
        result=extract_story('米娅幻化出长剑。她举起长剑。',judge=fake)
        self.assertEqual(result['status'],'ok')
        self.assertEqual([f['id'] for f in result['facts']],['F1'])
        self.assertEqual([f['id'] for f in result['discarded']],['F2'])
        self.assertEqual(result['rejected'],[])
        self.assertEqual(len(fake.calls),3)
        for messages in fake.calls[1:]:
            self.assertIn('candidate',json.loads(messages[1]['content']))
            self.assertNotIn('candidates',json.loads(messages[1]['content']))
            self.assertNotIn('story',json.loads(messages[1]['content']))

    def test_retained_fact_still_requires_subject_grounding(self):
        import json
        class Fake:
            device='cpu'
            last_generation={'seconds':1}
            def __init__(self):self.calls=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    return json.dumps({'facts':[{'subject':'米娅','predicate':'失明','type':'character_attribute','source_ids':['S2']}]})
                return '{"decision":"keep","reason":"永久身体变化"}'
        result=extract_story('米娅站在门外。她突然失明。',judge=Fake())
        self.assertNotEqual(result['status'],'ok')
        self.assertEqual(result['facts'],[])
        self.assertIn('主体依据',result['rejected'][0]['error'])

    def test_bad_single_decision_isolated_and_recorded(self):
        import json
        class Fake:
            device='cpu'
            last_generation={'seconds':1}
            def __init__(self):self.calls=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    return json.dumps({'facts':[
                        {'subject':'米娅','predicate':'失明','type':'character_attribute','source_ids':['S1']},
                        {'subject':'米娅','predicate':'失聪','type':'character_attribute','source_ids':['S2']}]})
                return '{"decision":"keep","reason":"身体变化"}' if self.calls==2 else '{"decision":"keep","discard":true}'
        result=extract_story('米娅失明。米娅失聪。',judge=Fake())
        self.assertEqual(result['status'],'partial')
        self.assertEqual([f['id'] for f in result['facts']],['F1'])
        self.assertEqual([f['id'] for f in result['pending_facts']],['F2'])
        self.assertEqual(len(result['timing']['calls']),3)
