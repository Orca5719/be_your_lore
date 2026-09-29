import json
import unittest
from story_judgement import check_facts, validate_answer, decode_reply


def report(status='ok'):
    fact=dict(id='F1',type='character_attribute',subject='雷',predicate='寄宿亚巴顿',object='左侧心脏',source_text='雷的左侧心脏寄宿亚巴顿。',context_text='',time=None)
    evidence=dict(id='C1',text='亚巴顿寄宿雷的左心脏。',file='angels.md',start_line=6,end_line=6,heading_path=['亚巴顿'],score=0.8)
    return dict(status=status,extraction=dict(status=status,story='不应进入判断的整段故事',facts=[fact],rejected=[]),items=[dict(status='ok',fact=fact,query='查询',evidence=[evidence])])


class StoryJudgementTests(unittest.TestCase):
    def test_only_uncertain_empty_citation_field_can_be_normalized(self):
        evidence=report()['items'][0]['evidence']
        raw='{"verdict":"不确定","reason":"资料没提到新能力","evidence_ids"}'
        answer=decode_reply(raw,evidence)
        self.assertEqual(answer['verdict'],'不确定')
        self.assertEqual(answer['evidence_ids'],[])
        self.assertTrue(answer['format_normalizations'])
        for verdict in ('一致','矛盾'):
            with self.assertRaises(ValueError):decode_reply(raw.replace('不确定',verdict),evidence)
        with self.assertRaises(ValueError):decode_reply('{"verdict":"不确定","reason":',evidence)

    def test_validation_requires_real_evidence_for_conclusion(self):
        evidence=report()['items'][0]['evidence']
        value=dict(verdict='一致',reason='左心脏宿主相符',evidence_ids=['C1'])
        self.assertEqual(validate_answer(value,evidence)['evidence'][0]['id'],'C1')
        for change in (dict(evidence_ids=[]),dict(evidence_ids=['C99']),dict(evidence_ids=['C1','C1']),dict(verdict='正确'),dict(reason='')):
            with self.assertRaises(ValueError):validate_answer(dict(value,**change),evidence)
        self.assertEqual(validate_answer(dict(value,verdict='不确定',evidence_ids=[]),evidence)['verdict'],'不确定')

    def test_one_fact_payload_and_program_bound_id(self):
        class Fake:
            last_generation={'seconds':1}
            def _generate(self,messages,**kwargs):
                self.messages=messages
                return json.dumps(dict(verdict='一致',reason='证据支持左心脏',evidence_ids=['C1']))
        judge=Fake()
        original=report()
        result=check_facts(original,judge)
        payload=json.loads(judge.messages[1]['content'])
        self.assertNotIn('story',payload)
        self.assertEqual(payload['fact']['id'],'F1')
        self.assertEqual(result['items'][0]['judgement']['verdict'],'一致')
        self.assertEqual(result['summary']['verdict'],'一致')
        self.assertNotIn('judgement',original['items'][0])

    def test_failure_is_separate_from_uncertain_and_later_calls_continue(self):
        class Fake:
            last_generation={'seconds':1}
            calls=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls<=2:return 'not JSON'
                return json.dumps(dict(verdict='矛盾',reason='右侧与左侧冲突',evidence_ids=['C1']))
        original=report()
        original['items'].append(dict(original['items'][0],fact=dict(original['items'][0]['fact'],id='F4')))
        result=check_facts(original,Fake())
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['items'][0]['judgement']['status'],'error')
        self.assertIsNone(result['items'][0]['judgement']['verdict'])
        self.assertEqual(result['items'][1]['judgement']['verdict'],'矛盾')
        self.assertEqual(result['summary']['verdict'],'矛盾')
        self.assertEqual(result['summary']['errors'],1)

    def test_invalid_json_retries_once_and_keeps_both_attempts(self):
        class Fake:
            last_generation={'seconds':1}
            calls=0
            def _generate(self,messages,**kwargs):
                self.calls+=1
                if self.calls==1:return '{"verdict":"矛盾","evidence_ids"}'
                return '{"verdict":"矛盾","reason":"右胸位置与左心脏宿主冲突","evidence_ids":["C1"]}'
        judge=Fake()
        result=check_facts(report(),judge)
        answer=result['items'][0]['judgement']
        self.assertEqual(result['status'],'ok')
        self.assertEqual(judge.calls,2)
        self.assertEqual(len(answer['attempts']),2)
        self.assertEqual(answer['attempts'][0]['status'],'error')
        self.assertEqual(answer['verdict'],'矛盾')
        self.assertEqual(len(result['judgement_calls']),2)

    def test_all_failures_display_incomplete_not_normal_uncertainty(self):
        class Fake:
            last_generation={}
            calls=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                return '{}'
        judge=Fake()
        result=check_facts(report(),judge)
        self.assertEqual(judge.calls,2)
        self.assertFalse(result['summary']['complete'])
        self.assertIn('检查未完成',result['summary']['display'])

    def test_no_evidence_is_uncertain_without_model_call(self):
        original=report()
        original['items'][0]['evidence']=[]
        result=check_facts(original,None)
        self.assertEqual(result['summary']['verdict'],'不确定')
        self.assertEqual(result['items'][0]['judgement']['status'],'ok')
        self.assertEqual(result['judgement_calls'],[])

    def test_partial_extraction_cannot_be_overall_consistent(self):
        class Fake:
            last_generation={}
            def _generate(self,*args,**kwargs):return '{"verdict":"一致","reason":"证据支持","evidence_ids":["C1"]}'
        result=check_facts(report('partial'),Fake())
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['summary']['verdict'],'不确定')

    def test_empty_and_retrieval_failure_do_not_call_model(self):
        original=report()
        original['items']=[]
        result=check_facts(original,None)
        self.assertIsNone(result['summary']['verdict'])
        original=report()
        original['status']='partial'
        original['items'][0]['status']='error'
        original['items'][0]['error']='检索失败'
        result=check_facts(original,None)
        self.assertEqual(result['items'][0]['judgement']['status'],'error')
        self.assertEqual(result['summary']['verdict'],'不确定')

    def test_cli_check_prints_judgement_and_evidence_sources(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout,redirect_stderr
        import io
        import story
        class Fake:
            last_generation={}
            def _generate(self,*args,**kwargs):return '{"verdict":"一致","reason":"左心脏设定支持","evidence_ids":["C1"]}'
        result=check_facts(report(),Fake())
        output=io.StringIO()
        with patch('story_judgement.check_story',return_value=result),redirect_stdout(output),redirect_stderr(io.StringIO()):
            code=story.main(['check','--text','雷左胸疼痛。','--device','cpu'])
        self.assertEqual(code,0)
        self.assertIn('判断：一致',output.getvalue())
        self.assertIn('引用证据：C1',output.getvalue())
        self.assertIn('angels.md',output.getvalue())

    def test_cli_json_keeps_errors_and_returns_nonzero(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout,redirect_stderr
        import io
        import story
        class Fake:
            last_generation={}
            def _generate(self,*args,**kwargs):return '{}'
        result=check_facts(report(),Fake())
        output=io.StringIO()
        with patch('story_judgement.check_story',return_value=result),redirect_stdout(output),redirect_stderr(io.StringIO()):
            code=story.main(['check','--text','雷左胸疼痛。','--json'])
        self.assertEqual(code,2)
        parsed=json.loads(output.getvalue())
        self.assertEqual(parsed['items'][0]['judgement']['status'],'error')
