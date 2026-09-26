import unittest
from agent_pipeline_benchmark import score_quality

class AgentPipelineBenchmarkTests(unittest.TestCase):
 def test_stage_metrics_have_distinct_denominators(self):
  gold=[dict(id='G1',expected_verdict='矛盾',acceptable_evidence_sets=[['C1']]),dict(id='G2',expected_verdict='一致',acceptable_evidence_sets=[['C2']]),dict(id='G3',expected_verdict='不确定',acceptable_evidence_sets=[])]
  cases=[dict(id='A',gold_facts=gold)]
  rows=[dict(case_id='A',result=dict(events=[{'id':'E1'},{'id':'E2'}],items=[dict(event_id='E1',status='ok',verdict='contradiction'),dict(event_id='E2',status='ok',verdict='contradiction')],retrieval={'items':[dict(event_id='E1',status='ok',evidence=[{'id':'C1'}]),dict(event_id='E2',status='ok',evidence=[])]}),seconds=3)]
  review={'A':{'G1':['E1'],'G2':['E2'],'G3':[]}}
  m=score_quality(cases,rows,review,5)
  self.assertEqual(m['recall_extraction']['hits'],2)
  self.assertEqual(m['recall_retrieval_at_k']['hits'],1)
  self.assertEqual(m['accuracy_judge']['eligible'],1)
  self.assertEqual(m['accuracy_judge']['correct'],1)
  self.assertEqual(m['end_to_end_conflict'],dict(tp=1,fp=1,fn=0,precision=.5,recall=1.0,f1=2/3))

 def test_missing_case_and_internal_false_conflict_count(self):
  cases=[dict(id='A',gold_facts=[dict(id='G1',expected_verdict='矛盾',acceptable_evidence_sets=[['C1']])])]
  rows=[dict(case_id='A',result=dict(events=[],items=[],story_check={'items':[{'status':'ok','verdict':'contradiction'}]}),seconds=1)]
  m=score_quality(cases,rows,{'A':{'G1':[]}},5)
  self.assertEqual(m['recall_extraction']['recall'],0)
  self.assertEqual(m['end_to_end_conflict']['fp'],1)
  self.assertEqual(m['end_to_end_conflict']['fn'],1)
