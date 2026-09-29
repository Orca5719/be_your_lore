import copy
import tempfile
from pathlib import Path
import unittest
from story_benchmark_metrics import score_results, performance_metrics
from story_benchmark import validate_dataset, run_cases


def fixtures():
    dataset={'cases':[
        {'id':'S1','gold_facts':[{'id':'G1','expected_verdict':'矛盾','acceptable_evidence_sets':[['C1']]},
                                {'id':'G2','expected_verdict':'一致','acceptable_evidence_sets':[['C2']]}]},
        {'id':'S2','gold_facts':[]}]}
    facts=[{'id':'F1'},{'id':'F2'}]
    rows=[{'case_id':'S1','seconds':2,'result':{'status':'partial','extraction':{'status':'ok','facts':facts},'items':[
        {'fact':facts[0],'status':'ok','evidence':[{'id':'C1'}],'judgement':{'status':'ok','verdict':'矛盾'}},
        {'fact':facts[1],'status':'ok','evidence':[],'judgement':{'status':'ok','verdict':'矛盾'}}]}},
        {'case_id':'S2','seconds':1,'result':{'status':'ok','extraction':{'status':'ok','facts':[]},'items':[]}}]
    review={'cases':[{'case_id':'S1','reviewed':True,'matches':[{'prediction_id':'F1','gold_id':'G1'}]},
                     {'case_id':'S2','reviewed':True,'matches':[]}]}
    return dataset,rows,review


class StoryBenchmarkTests(unittest.TestCase):
    def test_extraction_and_conflict_denominators_include_misses_and_false_positives(self):
        result=score_results(*fixtures())
        self.assertEqual(result['extraction']['tp'],1)
        self.assertEqual(result['extraction']['fp'],1)
        self.assertEqual(result['extraction']['fn'],1)
        self.assertEqual(result['conflict']['precision'],0.5)
        self.assertEqual(result['conflict']['recall'],1)
        self.assertEqual(result['classification']['accuracy'],0.5)
        self.assertEqual(result['evidence']['recall'],0.5)

    def test_failed_and_missing_cases_are_not_removed(self):
        dataset,rows,review=fixtures()
        rows=[]
        for case in review['cases']:case['matches']=[]
        result=score_results(dataset,rows,review)
        self.assertEqual(result['extraction']['fn'],2)
        self.assertEqual(result['conflict']['fn'],1)
        self.assertEqual(result['classification']['accuracy'],0)
        self.assertEqual(result['evidence']['recall'],0)
        self.assertEqual(result['daily']['clean_empty_rate'],0)
        self.assertEqual(result['processing']['failed_cases'],2)

    def test_review_is_required_and_one_to_one(self):
        dataset,rows,review=fixtures()
        review['cases'][0]['reviewed']=False
        with self.assertRaises(ValueError):score_results(dataset,rows,review)
        review['cases'][0]['reviewed']=True
        review['cases'][0]['matches'].append({'prediction_id':'F1','gold_id':'G2'})
        with self.assertRaises(ValueError):score_results(dataset,rows,review)

    def test_unknown_ids_and_duplicate_case_rows_rejected(self):
        dataset,rows,review=fixtures()
        review['cases'][0]['matches'][0]['gold_id']='G99'
        with self.assertRaises(ValueError):score_results(dataset,rows,review)
        dataset,rows,review=fixtures()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):score_results(dataset,rows,review)

    def test_zero_denominators_are_none(self):
        dataset={'cases':[{'id':'S1','gold_facts':[]}]}
        review={'cases':[{'case_id':'S1','reviewed':True,'matches':[]}]}
        result=score_results(dataset,[],review)
        self.assertIsNone(result['conflict']['precision'])
        self.assertIsNone(result['conflict']['recall'])
        self.assertIsNone(result['conflict']['f1'])
        self.assertIsNone(result['classification']['accuracy'])

    def test_failed_judgement_is_conflict_false_negative(self):
        dataset,rows,review=fixtures()
        rows[0]['result']['items'][0]['judgement']={'status':'error','verdict':None}
        result=score_results(dataset,rows,review)
        self.assertEqual(result['conflict']['tp'],0)
        self.assertEqual(result['conflict']['fn'],1)
        self.assertEqual(result['classification']['accuracy'],0)

    def test_performance_counts_all_calls_and_explicit_warmup(self):
        report={'model':'test','device':'cpu','warmup_count':1}
        rows=[{'seconds':2,'generations':[{'input_tokens':10,'ttft_ms':20,'decode_tokens':4,'decode_seconds':2}]},
              {'seconds':4,'generations':[{'input_tokens':30,'ttft_ms':40,'decode_tokens':6,'decode_seconds':1}]}]
        result=performance_metrics(report,rows)
        self.assertEqual(result['warm_total_latency_median_seconds'],3)
        self.assertEqual(result['avg_input_tokens'],20)
        self.assertAlmostEqual(result['decode_tokens_per_second'],10/3)
        self.assertIsNone(result['peak_allocated_gib'])

    def test_live_pilot_snapshot_and_tampered_anchor(self):
        import json
        root=Path(__file__).resolve().parents[1]
        dataset=json.loads((root/'evaluation/story_benchmark_pilot_4_v1.json').read_text(encoding='utf-8'))
        result=validate_dataset(dataset,root,root/'data/index')
        self.assertEqual(result['cases'],4)
        dataset['cases'][1]['gold_facts'][0]['source_anchor']['start']+=1
        with self.assertRaises(ValueError):validate_dataset(dataset,root,root/'data/index')

    def test_runner_saves_failures_and_continues(self):
        cases=[{'id':'A','story':'第一段'},{'id':'B','story':'第二段'}]
        def process(case):
            if case['id']=='A':raise RuntimeError('模拟模型失败')
            return {'status':'ok','extraction':{'status':'ok','facts':[]},'items':[]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.jsonl'
            rows=run_cases(cases,process,path)
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()),2)
            self.assertEqual(rows[0]['result']['status'],'error')
            self.assertEqual(rows[1]['result']['status'],'ok')

    def test_run_and_score_commands_require_bound_review_and_write_reports(self):
        import json
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from contextlib import redirect_stdout,redirect_stderr
        from story_benchmark import main,read_json,write_json
        fake_judge=SimpleNamespace(device='cpu',load_seconds=0.01,model_footprint_bytes=1024,
            model_cuda_allocated_bytes=None,torch=SimpleNamespace(get_num_threads=lambda:4))
        def empty_story(*args,**kwargs):
            return {'status':'ok','extraction':{'status':'ok','facts':[]},'items':[]}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run'
            with patch('qwen_judge.QwenJudge',return_value=fake_judge),patch('encoder.Encoder',return_value=SimpleNamespace(config={})),patch('retrieval.Retriever',return_value=SimpleNamespace(metadata={'config':{}})),patch('story_judgement.check_story',side_effect=empty_story),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(['run','--device','cpu','--output',str(output)]),0)
            self.assertEqual(read_json(output/'report.json')['saved_cases'],4)
            self.assertTrue((output/'performance.json').exists())
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(['score',str(output)]),2)
            review=read_json(output/'review.json')
            for case in review['cases']:case['reviewed']=True
            write_json(output/'review.json',review)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(['score',str(output)]),0)
            metrics=read_json(output/'metrics.json')
            self.assertEqual(metrics['extraction']['fn'],5)
            self.assertEqual(metrics['conflict']['fn'],1)
            review['raw_sha256']='wrong-run'
            write_json(output/'review.json',review)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(['score',str(output)]),2)

    def test_multisource_evidence_requires_whole_set(self):
        dataset,rows,review=fixtures()
        dataset['cases'][0]['gold_facts'][0]['acceptable_evidence_sets']=[['C1','C3']]
        self.assertEqual(score_results(dataset,rows,review)['evidence']['hits'],0)
        rows[0]['result']['items'][0]['evidence'].append({'id':'C3'})
        self.assertEqual(score_results(dataset,rows,review)['evidence']['hits'],1)
