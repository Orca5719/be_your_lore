import copy
import unittest
from unittest.mock import patch
from agent_pipeline.judge import judge_events
from agent_pipeline.retrieval import retrieve_events
from test_agent_pipeline_retrieval import filtered,FakeRetriever
from test_agent_pipeline_filtering import FakeLLM


def retrieved(count=1,status='ok'):
    return retrieve_events(filtered(count,status),retriever=FakeRetriever())


def answer(verdict='consistent',eid='L1',quote='设定原文'):
    return dict(verdict=verdict,citations=[dict(evidence_id=eid,quote=quote)],reason='原文支持对应命题',assessment=dict(same_subject=True,evidence_applicable=True,relation='direct_conflict' if verdict=='contradiction' else 'direct_support',assumptions=[]))


class JudgeTests(unittest.TestCase):
    def test_uncertain_reason_is_rendered_from_code_not_invented_body_side(self):
        value=answer('uncertain');value.update(reason='右臂具有某种液化性质',uncertainty_code='ambiguous_reference')
        result=judge_events(retrieved(),llm=FakeLLM([value,value]))
        item=result['items'][0]
        self.assertEqual(item['status'],'ok')
        self.assertEqual(item['model_reason'],value['reason'])
        self.assertNotIn('右臂',item['reason'])
        self.assertEqual(item['reason_origin'],'program_uncertainty_code')

    def test_unknown_uncertainty_code_is_not_accepted_as_new_fact(self):
        value=answer('uncertain');value['uncertainty_code']='secret_right_arm'
        result=judge_events(retrieved(),llm=FakeLLM([value,value]))
        self.assertEqual(result['status'],'error')
    def test_citations_bind_to_canonical_ids_and_input_unchanged(self):
        report=retrieved();before=copy.deepcopy(report)
        result=judge_events(report,llm=FakeLLM([answer()]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['items'][0]['citations'][0]['chunk_id'],'C1')
        self.assertEqual(report,before)

    def test_fabricated_citation_retry_preserves_failed_raw(self):
        result=judge_events(retrieved(),llm=FakeLLM([answer(eid='L99'),answer('contradiction')]))
        self.assertEqual(result['items'][0]['verdict'],'contradiction')
        self.assertEqual(len(result['items'][0]['call']['attempts']),2)

    def test_quote_not_in_evidence_cannot_pass(self):
        result=judge_events(retrieved(),llm=FakeLLM([answer(quote='发明证据'),answer(quote='发明证据')]))
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['items'][0]['status'],'error')
        self.assertEqual(result['items'][0]['verdict'],'uncertain')

    def test_no_evidence_uncertain_without_loading_qwen(self):
        report=retrieved();report['items'][0]['evidence']=[]
        result=judge_events(report)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['items'][0]['origin'],'program_no_evidence')
        self.assertFalse(result['model_loaded_this_request'])

    def test_normal_uncertainty_is_not_processing_failure(self):
        value=answer('uncertain');value.update(citations=[],assessment=dict(same_subject=None,evidence_applicable=None,relation='insufficient',assumptions=[]))
        result=judge_events(retrieved(),llm=FakeLLM([value]))
        self.assertEqual(result['status'],'ok')
        self.assertTrue(result['processing_complete'])

    def test_runtime_failure_is_not_retried_and_other_event_continues(self):
        model=FakeLLM([RuntimeError('GPU failure'),answer()])
        result=judge_events(retrieved(2),llm=model)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['items'][1]['status'],'ok')
        self.assertEqual(len(model.messages),2)

    def test_upstream_partial_not_masked(self):
        result=judge_events(retrieved(1,'partial'),llm=FakeLLM([answer()]))
        self.assertEqual(result['status'],'partial')

    def test_understanding_failure_details_reach_judge_summary(self):
        report=retrieved(1,'partial')
        reasons={'unresolved_candidates':[{'candidate_id':'W1C2','error':'主体错误'}]}
        report['filtering']['understanding']={'stage':'understanding','status':'partial','failure_reasons':reasons}
        result=judge_events(report,llm=FakeLLM([answer()]))
        self.assertEqual(result['failure_reasons']['upstream_stages']['understanding'],reasons)

    def test_invalid_event_partition_rejected_before_loading(self):
        report=retrieved();report['items'][0]['event_id']='E99'
        with self.assertRaises(ValueError):judge_events(report)

    def test_conclusive_answer_requires_evidence(self):
        invalid=answer();invalid.update(citations=[],reason='没有依据')
        result=judge_events(retrieved(),llm=FakeLLM([invalid,invalid]))
        self.assertEqual(result['status'],'error')

    def test_empty_input_no_model_and_no_story_verdict(self):
        result=judge_events(retrieved(0))
        self.assertEqual(result['items'],[])
        self.assertFalse(result['model_loaded_this_request'])
        self.assertNotIn('overall_verdict',result)

    def test_model_load_failure_keeps_provenance_and_does_not_repeat_load(self):
        with patch('qwen_judge.QwenJudge',side_effect=RuntimeError('模型加载失败')) as loader:
            result=judge_events(retrieved(2))
        self.assertEqual(loader.call_count,1)
        self.assertEqual(result['status'],'error')
        self.assertEqual(len(result['items']),2)
        self.assertEqual(result['retrieval']['stage'],'retrieval')

    def test_nonactual_content_does_not_become_objective_conflict(self):
        for modality in ('dream','belief','speech','plan','inferred'):
            report=retrieved();event=report['events'][0]
            event.update(modality=modality,explicit=modality!='inferred')
            model=FakeLLM([answer('contradiction')])
            result=judge_events(report,llm=model)
            self.assertEqual(result['items'][0]['verdict'],'uncertain')
            self.assertEqual(result['items'][0]['origin'],'program_nonactual_scope')
            self.assertEqual(model.messages,[])

    def test_forged_success_cannot_hide_nested_partial(self):
        report=retrieved(1,'partial');report['status']='ok'
        with self.assertRaises(ValueError):judge_events(report,llm=FakeLLM([answer()]))

    def test_missing_pending_id_field_rejected(self):
        report=retrieved();report.pop('pending_event_ids')
        with self.assertRaises(ValueError):judge_events(report,llm=FakeLLM([answer()]))

    def test_assumptions_and_unknown_scope_cannot_yield_conclusive_verdict(self):
        for change in ({'assumptions':['假定未指明的手臂是右臂']},{'evidence_applicable':False},{'same_subject':None},{'relation':'insufficient'}):
            value=answer('contradiction');value['assessment'].update(change)
            result=judge_events(retrieved(),llm=FakeLLM([value]))
            self.assertEqual(result['items'][0]['verdict'],'uncertain')
            self.assertEqual(result['items'][0]['model_verdict'],'contradiction')
            self.assertEqual(result['status'],'ok')

    def test_judge_receives_scene_and_lore_heading_without_similarity(self):
        report=retrieved();report['filtering']['understanding']={'text':'实验室的主机旁，屏幕变化。'}
        report['items'][0]['evidence'][0]['heading_path']=['人物','广场新闻']
        model=FakeLLM([answer()]);judge_events(report,llm=model)
        import json
        payload=json.loads(model.messages[0][1]['content'])
        self.assertEqual(payload['story_context'],'实验室的主机旁，屏幕变化。')
        self.assertEqual(payload['lore'][0]['heading_path'],['人物','广场新闻'])
        self.assertNotIn('score',payload['lore'][0])
