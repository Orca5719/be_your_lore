import tempfile
import unittest
from pathlib import Path
from agent_pipeline.freshness import validate_sources
from agent_pipeline.story_check import check_story
from agent_pipeline.judge import judge_events
from agent_pipeline.report import build_report,render_author_report
from test_agent_pipeline_judge import retrieved
from test_agent_pipeline_filtering import FakeLLM,upstream

class StructuralFixTests(unittest.TestCase):
    def test_nonactual_without_evidence_never_proposes_canon(self):
        r=retrieved();r['events'][0]['modality']='dream';r['items'][0]['evidence']=[]
        d=judge_events(r)
        self.assertEqual(d['items'][0]['origin'],'program_nonactual_scope')
        self.assertEqual(build_report(d)['report']['possible_additions'],[])

    def test_source_add_edit_delete_detected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'a.md';p.write_text('old',encoding='utf-8')
            m={'lore_directory':folder,'sources':{'a.md':hashlib.sha256(p.read_bytes()).hexdigest()}}
            validate_sources(m)
            p.write_text('new',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'重建'):validate_sources(m)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'重建'):validate_sources(m)
            p.write_text('old',encoding='utf-8');(Path(folder)/'b.txt').write_text('added')
            with self.assertRaisesRegex(ValueError,'重建'):validate_sources(m)

    def test_story_conflict_binds_both_original_events(self):
        u=upstream();a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':True,'assumptions':[],'reason':'同时互斥','citations':[{'event_id':'E1','quote':'原文'},{'event_id':'E2','quote':'原文'}]}]}
        result=check_story(u,llm=FakeLLM([a]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['items'][0]['event_ids'],['E1','E2'])

    def test_different_scope_is_not_promoted_to_conflict(self):
        a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':False,'assumptions':[],'reason':'前后变化','citations':[{'event_id':'E1','quote':'原文'},{'event_id':'E2','quote':'原文'}]}]}
        result=check_story(upstream(),llm=FakeLLM([a]))
        self.assertEqual(result['items'][0]['verdict'],'uncertain')

    def test_non_conflict_drops_unneeded_bad_citations_without_failure(self):
        a={'checks':[{'pair_id':'P1','verdict':'no_conflict','same_subject':True,'same_scope':True,'assumptions':[],'reason':'并列事实','citations':[{'event_id':'E1','quote':'标点被模型改坏。'}]}]}
        result=check_story(upstream(),llm=FakeLLM([a]))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['items'][0]['citations'],[])
        self.assertEqual(result['items'][0]['wire_normalizations'],['non_contradiction_citations_removed'])

    def test_contradiction_still_rejects_bad_citations(self):
        a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':True,'assumptions':[],'reason':'冲突','citations':[{'event_id':'E1','quote':'标点被模型改坏。'}]}]}
        result=check_story(upstream(),llm=FakeLLM([a,a,a]))
        self.assertEqual(result['status'],'partial')

    def test_bad_citation_is_failure_not_no_conflict(self):
        a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':True,'assumptions':[],'reason':'冲突','citations':[{'event_id':'E1','quote':'编造'}]}]}
        result=check_story(upstream(),llm=FakeLLM([a,a,a]))
        self.assertEqual(result['status'],'partial')

    def test_valid_story_rows_survive_and_only_bad_pair_is_retried(self):
        u=upstream(3)
        good=lambda pid:{'pair_id':pid,'verdict':'no_conflict','same_subject':True,'same_scope':True,'assumptions':[],'reason':'没有冲突','citations':[]}
        bad=dict(good('P1'),verdict='contradiction',citations=[{'event_id':'E3','quote':'原文'}])
        model=FakeLLM([{'checks':[bad,good('P2'),good('P3')]},{'checks':[good('P1')]}])
        result=check_story(u,llm=model)
        self.assertEqual(result['status'],'ok')
        self.assertEqual([item['status'] for item in result['items']],['ok','ok','ok'])
        self.assertEqual(len(model.messages),2)
        repair_payload=__import__('json').loads(model.messages[1][1]['content'])
        self.assertEqual([p['pair_id'] for p in repair_payload['pairs']],['P1'])

    def test_nonactual_excluded_and_pair_limit_reported(self):
        u=upstream();u['events'][0]['modality']='dream'
        self.assertEqual(check_story(u)['pair_count'],0)
        r=check_story(upstream(17),max_pairs=0)
        self.assertEqual(r['status'],'partial')
        self.assertGreater(r['unprocessed_pair_count'],0)

    def test_story_conflict_changes_summary_and_is_printed(self):
        r=retrieved(2);u={'events':r['events'],'status':'ok'}
        a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':True,'assumptions':[],'reason':'两个原文断言不能同时成立','citations':[{'event_id':'E1','quote':'原文'},{'event_id':'E2','quote':'原文'}]}]}
        d=judge_events(r,llm=FakeLLM([__import__('test_agent_pipeline_judge').answer('uncertain')]*2))
        d['story_check']=check_story(u,llm=FakeLLM([a]))
        report=build_report(d)
        self.assertEqual(report['summary']['verdict'],'contradiction')
        self.assertIn('段内矛盾',render_author_report(report))
        self.assertEqual(len(report['report']['story_consistency']['contradictions']),1)

    def test_historical_report_does_not_claim_story_check(self):
        d=judge_events(retrieved(),llm=FakeLLM([__import__('test_agent_pipeline_judge').answer()]))
        self.assertEqual(build_report(d)['report']['story_consistency']['status'],'not_run')

    def test_tampered_story_citation_rejected(self):
        r=retrieved(2);d=judge_events(r,llm=FakeLLM([__import__('test_agent_pipeline_judge').answer()]*2))
        a={'checks':[{'pair_id':'P1','verdict':'contradiction','same_subject':True,'same_scope':True,'assumptions':[],'reason':'冲突','citations':[{'event_id':'E1','quote':'原文'},{'event_id':'E2','quote':'原文'}]}]}
        d['story_check']=check_story({'events':r['events'],'status':'ok'},llm=FakeLLM([a]))
        d['story_check']['items'][0]['citations'][0]['quote']='伪造'
        with self.assertRaises(ValueError):build_report(d)

    def test_stale_index_fails_before_encoder_loading(self):
        from unittest.mock import patch
        from agent_pipeline.retrieval import retrieve_events
        from test_agent_pipeline_retrieval import filtered
        with patch('index_store.load_index',return_value=(None,{'lore_directory':'missing','sources':{}})),patch('encoder.Encoder') as enc:
            r=retrieve_events(filtered(1))
        enc.assert_not_called()
        self.assertEqual(r['status'],'error')
        self.assertIn('重建',r['failure_reasons']['load_error'])

    def test_historical_no_evidence_nonactual_also_excluded(self):
        r=retrieved();r['events'][0]['modality']='dream';r['items'][0]['evidence']=[]
        d=judge_events(r);d['items'][0]['origin']='program_no_evidence'
        self.assertEqual(build_report(d)['report']['possible_additions'],[])

    def test_contiguous_source_quote_allowed_but_gap_rejected(self):
        from agent_pipeline.story_check import citation_texts
        e={'sources':[{'text':'甲在','start':0,'end':2},{'text':'屋内。','start':2,'end':5}], 'contexts':[]}
        self.assertIn('甲在屋内。',citation_texts(e))
        e['sources'][1].update(start=3,end=6)
        self.assertNotIn('甲在屋内。',citation_texts(e))

    def test_nonobject_story_row_is_contained_failure(self):
        r=check_story(upstream(),llm=FakeLLM([{'checks':[None]},{'checks':[None]},{'checks':[None]}]))
        self.assertEqual(r['status'],'partial')
        self.assertEqual(len(r['calls']),2)

    def test_nonobject_saved_story_row_rejected_cleanly(self):
        from agent_pipeline.story_check import validate_story_result
        with self.assertRaises(ValueError):
            validate_story_result(dict(stage='story_check',status='ok',pair_count=1,unprocessed_pair_count=0,items=[None]),upstream()['events'])

    def test_story_check_reuses_filtering_selected_events(self):
        from agent_pipeline.filtering import filter_events
        from test_agent_pipeline_filtering import decision
        f=filter_events(upstream(3),llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2'),decision('E3','ignore')]}]))
        r=check_story(f)
        self.assertEqual(r['candidate_event_ids'],['E2'])
        self.assertEqual(r['pair_count'],0)
        self.assertEqual(r['filtered_out_event_count'],2)

    def test_support_only_is_context_not_story_candidate(self):
        from agent_pipeline.filtering import filter_events
        from test_agent_pipeline_filtering import decision
        u=upstream(2);u['events'][1]['context_ids']=['S1']
        f=filter_events(u,llm=FakeLLM([{'decisions':[decision('E1','ignore'),decision('E2')]}]))
        self.assertTrue(f['decisions'][0]['support_only'])
        self.assertEqual(check_story(f)['candidate_event_ids'],['E2'])
