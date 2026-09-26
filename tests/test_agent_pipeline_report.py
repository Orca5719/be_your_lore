import copy
import unittest
from agent_pipeline.report import build_report
from agent_pipeline.judge import judge_events
from test_agent_pipeline_judge import retrieved,answer
from test_agent_pipeline_filtering import FakeLLM
from test_agent_pipeline_filtering import upstream,decision
from agent_pipeline.filtering import filter_events
from agent_pipeline.retrieval import retrieve_events
from test_agent_pipeline_retrieval import FakeRetriever


def judged(count=1,verdict='consistent',status='ok'):
    return judge_events(retrieved(count,status),llm=FakeLLM([answer(verdict) for _ in range(count)]))


class ReportTests(unittest.TestCase):
    def test_complete_report_has_overview_sections_and_confirmation_topics(self):
        report=build_report(judged(verdict='uncertain'))
        presentation=report['report']
        self.assertIn('1',presentation['overview'])
        self.assertEqual(presentation['uncertainties'][0]['finding_ids'],['R1'])
        self.assertEqual(presentation['confirmation_topics'][0]['event_ids'],['E1'])
        self.assertFalse(presentation['confirmation_topics'][0]['saved'])

    def test_report_sections_do_not_change_conflict_or_hide_partial(self):
        value=judge_events(retrieved(2),llm=FakeLLM([RuntimeError('OOM'),answer('contradiction')]))
        report=build_report(value)
        self.assertEqual(report['report']['contradictions'][0]['finding_ids'],['R1'])
        self.assertIn('未完整',report['report']['overview'])
        self.assertTrue(report['report']['processing_issues'])
    def test_contradiction_has_evidence_source_and_unchanged_input(self):
        value=judged(verdict='contradiction');before=copy.deepcopy(value)
        report=build_report(value)
        self.assertEqual(report['summary']['verdict'],'contradiction')
        self.assertEqual(report['findings'][0]['event_ids'],['E1'])
        self.assertEqual(report['evidence'][0]['file'],'characters.md')
        self.assertEqual(report['evidence'][0]['start_line'],6)
        self.assertEqual(report['judge'],value)
        self.assertEqual(value,before)

    def test_uncertainty_is_success_and_no_story_claim(self):
        report=build_report(judged(verdict='uncertain'))
        self.assertEqual(report['status'],'ok')
        self.assertEqual(report['summary']['verdict'],'uncertain')
        self.assertEqual(report['summary']['scope'],'checked_events')

    def test_same_proposition_source_and_evidence_merge_but_keep_ids(self):
        value=judged(2)
        for layer in [value,value['retrieval'],value['retrieval']['filtering']]:
            layer['events'][1].update({k:copy.deepcopy(v) for k,v in layer['events'][0].items() if k!='id'})
        for item in value['items']:item['event']=copy.deepcopy(value['events'][int(item['event_id'][1:])-1])
        for item in value['retrieval']['items']:item['event']=copy.deepcopy(value['events'][int(item['event_id'][1:])-1])
        value['retrieval']['filtering']['selected_events']=copy.deepcopy(value['events'])
        report=build_report(value)
        self.assertEqual(len(report['findings']),1)
        self.assertEqual(report['findings'][0]['event_ids'],['E1','E2'])
        self.assertEqual(report['summary']['counts']['consistent'],2)

    def test_same_evidence_different_events_not_merged(self):
        report=build_report(judged(2))
        self.assertEqual(len(report['findings']),2)
        self.assertEqual(len(report['evidence']),1)

    def test_same_text_without_offsets_but_different_source_ids_not_merged(self):
        value=judged(2)
        for layer in [value,value['retrieval'],value['retrieval']['filtering']]:
            layer['events'][1].update({k:copy.deepcopy(v) for k,v in layer['events'][0].items() if k not in ('id','source_ids')})
            layer['events'][1]['source_ids']=['S2']
        for layer in [value,value['retrieval']]:
            for i,item in enumerate(layer['items']):item['event']=copy.deepcopy(value['events'][i])
        value['retrieval']['filtering']['selected_events']=copy.deepcopy(value['events'])
        self.assertEqual(len(build_report(value)['findings']),2)

    def test_uncertain_with_different_candidates_not_merged(self):
        value=judged(2,verdict='uncertain')
        for layer in [value,value['retrieval'],value['retrieval']['filtering']]:
            layer['events'][1].update({k:copy.deepcopy(v) for k,v in layer['events'][0].items() if k!='id'})
        for layer in [value,value['retrieval']]:
            for i,item in enumerate(layer['items']):item['event']=copy.deepcopy(value['events'][i])
            layer['items'][1]['evidence'][0]['id']='C2'
        for item in value['items']:item['citations']=[]
        value['retrieval']['filtering']['selected_events']=copy.deepcopy(value['events'])
        self.assertEqual(len(build_report(value)['findings']),2)

    def test_no_source_identifiers_or_offsets_never_merges_distinct_events(self):
        value=judged(2)
        for layer in [value,value['retrieval'],value['retrieval']['filtering']]:
            layer['events'][1].update({k:copy.deepcopy(v) for k,v in layer['events'][0].items() if k!='id'})
            for event in layer['events']:event.pop('source_ids',None)
        for layer in [value,value['retrieval']]:
            for i,item in enumerate(layer['items']):item['event']=copy.deepcopy(value['events'][i])
        value['retrieval']['filtering']['selected_events']=copy.deepcopy(value['events'])
        self.assertEqual(len(build_report(value)['findings']),2)

    def test_upstream_partial_is_not_hidden(self):
        report=build_report(judged(status='partial'))
        self.assertEqual(report['status'],'partial')
        self.assertFalse(report['processing_complete'])
        self.assertEqual(report['summary']['verdict'],'uncertain')

    def test_error_is_separate_from_normal_uncertain(self):
        value=judge_events(retrieved(2),llm=FakeLLM([RuntimeError('OOM'),answer()]))
        report=build_report(value)
        self.assertEqual(report['summary']['counts']['failed'],1)
        self.assertEqual(report['summary']['counts']['uncertain'],0)
        self.assertEqual(report['failed_items'][0]['event_id'],'E1')

    def test_empty_report_does_not_assert_consistency(self):
        report=build_report(judged(0))
        self.assertEqual(report['summary']['verdict'],'uncertain')
        self.assertEqual(report['status'],'ok')

    def test_pending_and_ignored_partitions_are_preserved(self):
        filtered=filter_events(upstream(2),llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2','review')]}]))
        value=judge_events(retrieve_events(filtered,retriever=FakeRetriever()))
        report=build_report(value)
        self.assertEqual(report['pending_events'][0]['id'],'E2')
        self.assertEqual(report['report']['processing_issues'][0]['fact'],report['pending_events'][0]['event'])
        self.assertEqual(report['ignored_events'][0]['id'],'E1')
        self.assertEqual(report['summary']['counts']['pending'],1)
        self.assertEqual(report['status'],'partial')

    def test_top_level_error_retains_unprocessed_events(self):
        value=judge_events(retrieved(1,'error'))
        report=build_report(value)
        self.assertEqual(report['status'],'error')
        self.assertEqual(report['summary']['counts']['unprocessed'],1)
        self.assertEqual(report['unprocessed_events'][0]['id'],'E1')

    def test_confirmed_conflict_survives_another_event_failure(self):
        value=judge_events(retrieved(2),llm=FakeLLM([RuntimeError('OOM'),answer('contradiction')]))
        report=build_report(value)
        self.assertEqual(report['status'],'partial')
        self.assertEqual(report['summary']['verdict'],'contradiction')
        self.assertEqual(report['summary']['counts']['failed'],1)

    def test_bad_canonical_quote_and_missing_item_rejected(self):
        for change in ('quote','missing','duplicate'):
            value=judged()
            if change=='quote':value['items'][0]['citations'][0]['quote']='伪造'
            elif change=='missing':value['items']=[]
            else:value['items'].append(copy.deepcopy(value['items'][0]))
            with self.assertRaises(ValueError):build_report(value)

    def test_forged_success_does_not_hide_upstream_partial(self):
        value=judged(status='partial');value['status']='ok'
        with self.assertRaises(ValueError):build_report(value)
