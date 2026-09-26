"""Provisional quality/latency benchmark for the current five-stage agent pipeline."""
import argparse,hashlib,json,math,statistics,time,uuid
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATASET=ROOT/'evaluation/story_benchmark_pilot_4_v1.json'

def ratio(a,b):return a/b if b else None
def prf(tp,fp,fn):
 p=ratio(tp,tp+fp);r=ratio(tp,tp+fn)
 return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=ratio(2*tp,2*tp+fp+fn))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def score_quality(cases,run_rows,review,k=5):
 byrow={r['case_id']:r for r in run_rows};reviewed=review
 extraction_hits=extraction_total=retrieval_hits=retrieval_total=judge_correct=judge_eligible=0
 tp=fp=fn=0;confusion={}
 for case in cases:
  cid=case['id'];result=byrow.get(cid,{}).get('result',{});mapping=reviewed.get(cid,{})
  events={e['id']:e for e in result.get('events',[])}
  judge=result.get('judge',result);items={i['event_id']:i for i in judge.get('items',[]) if i.get('status')!='support_only'}
  retrieval={i['event_id']:i for i in judge.get('retrieval',{}).get('items',[]) if i.get('status')!='support_only'}
  conflict_gold={g['id'] for g in case['gold_facts'] if g['expected_verdict']=='矛盾'}
  covered_conflicts=set();positive_preds=set()
  for eid,item in items.items():
   if item.get('status')=='ok' and item.get('verdict')=='contradiction':positive_preds.add(eid)
  for gold in case['gold_facts']:
   gid=gold['id'];mapped=[eid for eid in mapping.get(gid,[]) if eid in events]
   extraction_total+=1;extraction_hits+=bool(mapped)
   alternatives=gold.get('acceptable_evidence_sets',[]);retrieved_ok=False
   if alternatives:
    retrieval_total+=1
    for eid in mapped:
     row=retrieval.get(eid,{});found={x['id'] for x in row.get('evidence',[])[:k]} if row.get('status')=='ok' else set()
     if any(set(group)<=found for group in alternatives):retrieved_ok=True;break
    retrieval_hits+=retrieved_ok
   retrieval_ready=any(retrieval.get(eid,{}).get('status')=='ok' for eid in mapped)
   eligible=bool(mapped) and retrieval_ready and (not alternatives or retrieved_ok)
   if eligible:
    judge_eligible+=1
    predicted={items[eid].get('verdict') for eid in mapped if items.get(eid,{}).get('status')=='ok'}
    expected={'一致':'consistent','矛盾':'contradiction','不确定':'uncertain'}[gold['expected_verdict']]
    ok=expected in predicted;judge_correct+=ok
    confusion.setdefault(expected,Counter())[next(iter(predicted)) if len(predicted)==1 else 'mixed_or_missing']+=1
   if gid in conflict_gold and any(eid in positive_preds for eid in mapped):covered_conflicts.add(gid)
  reverse={eid:{gid for gid,ids in mapping.items() if eid in ids} for eid in positive_preds}
  tp+=len(covered_conflicts);fn+=len(conflict_gold-covered_conflicts)
  fp+=sum(not bool(gids&conflict_gold) for gids in reverse.values())
  story=judge.get('story_check',result.get('story_check',{}))
  fp+=sum(i.get('status')=='ok' and i.get('verdict')=='contradiction' for i in story.get('items',[]))
 return dict(recall_extraction=dict(hits=extraction_hits,gold_total=extraction_total,recall=ratio(extraction_hits,extraction_total)),
  recall_retrieval_at_k=dict(hits=retrieval_hits,gold_total=retrieval_total,k=k,recall=ratio(retrieval_hits,retrieval_total)),
  accuracy_judge=dict(correct=judge_correct,eligible=judge_eligible,accuracy=ratio(judge_correct,judge_eligible),denominator='Gold facts with a semantic event match whose matched event completed retrieval; evidence-bearing gold additionally requires an acceptable evidence set within Top-K.',confusion={k:dict(v) for k,v in confusion.items()}),
  end_to_end_conflict=prf(tp,fp,fn),scope='Provisional 4-story draft gold and assistant-reviewed semantic coverage. Missing extraction/retrieval/judgement remains FN end-to-end.')

def source_snapshot():
 names=['agent_pipeline/understanding.py','agent_pipeline/filtering.py','agent_pipeline/retrieval.py','agent_pipeline/judge.py','agent_pipeline/story_check.py','agent_pipeline/report.py','agent_pipeline_benchmark.py','agent_pipeline/prompts/understanding_v8.txt','agent_pipeline/prompts/filtering_v7.txt','agent_pipeline/prompts/judge_v6.txt','agent_pipeline/prompts/story_check_v1.txt']
 return {n:digest(ROOT/n) for n in names}

def process_story(text,llm,retriever,top_k,progress=None):
 from agent_pipeline.understanding import understand
 from agent_pipeline.filtering import filter_events
 from agent_pipeline.retrieval import retrieve_events
 from agent_pipeline.judge import judge_events
 from agent_pipeline.story_check import check_story
 from agent_pipeline.report import build_report
 u=understand(text,llm=llm,progress=progress);f=filter_events(u,llm=llm,progress=progress);r=retrieve_events(f,retriever=retriever,k=top_k,progress=progress);j=judge_events(r,llm=llm,progress=progress);j['story_check']=check_story(f,llm=llm,progress=progress)
 if j['story_check']['status']!='ok':
  if j['status']=='ok':j['status']='partial'
  j['processing_complete']=False;j['failure_reasons']['story_check']='段内检查未完整完成'
 return build_report(j)

def run(args):
 from story_benchmark import validate_dataset
 data=read(args.dataset);validation=validate_dataset(data,ROOT,args.index)
 out=args.output or ROOT/'agent_pipeline/reports'/('pipeline_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True,exist_ok=False)
 (out/'dataset.json').write_bytes(args.dataset.read_bytes());snapshot=source_snapshot()
 from qwen_judge import QwenJudge,MODEL,REVISION
 from encoder import Encoder
 from retrieval import Retriever
 llm=QwenJudge(args.device);t=time.perf_counter();encoder=Encoder(device=llm.device,offline=True,precision='float32');retriever=Retriever(args.index,encoder);encoder_seconds=time.perf_counter()-t
 warm=[]
 def progress(step):print('  '+step['purpose']+' window '+str(step['window_id']),flush=True)
 for _ in range(args.warmup):
  t=time.perf_counter()
  try:
   print('warmup begin',flush=True);result=process_story('雷拥有两颗心脏。',llm,retriever,args.top_k,progress);warm.append(dict(seconds=time.perf_counter()-t,result=result))
  except (ValueError,OSError,RuntimeError) as e:warm.append(dict(seconds=time.perf_counter()-t,error=str(e)))
 write(out/'warmup.json',warm)
 runrows=[]
 with (out/'raw.jsonl').open('x',encoding='utf-8') as stream:
  for i,case in enumerate(data['cases'],1):
   print(f'{i}/{len(data["cases"])} {case["id"]} begin',flush=True)
   if llm.device=='cuda':llm.torch.cuda.synchronize();llm.torch.cuda.reset_peak_memory_stats()
   t=time.perf_counter()
   try:result=process_story(case['story'],llm,retriever,args.top_k,progress)
   except (ValueError,OSError,RuntimeError) as e:result=dict(stage='report',status='error',error=str(e),events=[],items=[])
   if llm.device=='cuda':llm.torch.cuda.synchronize()
   row=dict(case_id=case['id'],seconds=time.perf_counter()-t,result=result)
   if llm.device=='cuda':row.update(peak_allocated_bytes=llm.torch.cuda.max_memory_allocated(),peak_reserved_bytes=llm.torch.cuda.max_memory_reserved())
   stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();runrows.append(row)
   print(f'{i}/{len(data["cases"])} {case["id"]} {result.get("status")} {row["seconds"]:.2f}s',flush=True)
 seconds=[r['seconds'] for r in runrows]
 report=dict(name='Agent Pipeline Quality Benchmark',status='completed',dataset_status=data['status'],validation=validation,top_k=args.top_k,device=llm.device,model=MODEL,revision=REVISION,model_load_seconds=llm.load_seconds,encoder_index_load_seconds=encoder_seconds,
  cases=len(runrows),case_status_counts=dict(Counter(r['result'].get('status') for r in runrows)),latency_end_to_end=dict(median_seconds=statistics.median(seconds),p95_seconds=sorted(seconds)[max(0,math.ceil(.95*len(seconds))-1)],mean_seconds=statistics.mean(seconds),per_case_seconds={r['case_id']:r['seconds'] for r in runrows}),
  peak_allocated_gib=max((r.get('peak_allocated_bytes',0) for r in runrows),default=0)/2**30,peak_reserved_gib=max((r.get('peak_reserved_bytes',0) for r in runrows),default=0)/2**30,source_sha256=snapshot,snapshot_stable=snapshot==source_snapshot(),raw_sha256=digest(out/'raw.jsonl'),scoring_status='pending_semantic_review')
 write(out/'report.json',report);write(out/'review.json',dict(status='pending',cases={c['id']:{g['id']:[] for g in c['gold_facts']} for c in data['cases']}))
 print('RESULT_DIR='+str(out));return 0

def score(args):
 data=read(args.directory/'dataset.json');runrows=rows(args.directory/'raw.jsonl');review=read(args.review or args.directory/'review.json')
 if review.get('status')!='reviewed':raise ValueError('review.json尚未完成语义覆盖审核')
 metrics=score_quality(data['cases'],runrows,review['cases'],read(args.directory/'report.json')['top_k'])
 metrics['scoring_provenance']=dict(dataset_sha256=digest(args.directory/'dataset.json'),raw_sha256=digest(args.directory/'raw.jsonl'),review_sha256=digest(args.review or args.directory/'review.json'),scorer_sha256=digest(__file__))
 write(args.directory/'quality.json',metrics)
 report=read(args.directory/'report.json');report['scoring_status']='scored_provisional';report['quality']=metrics;write(args.directory/'report.json',report)
 print(json.dumps(dict(quality=metrics,latency_end_to_end=report['latency_end_to_end']),ensure_ascii=False,indent=2));return 0

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True);r=sp.add_parser('run');r.add_argument('--dataset',type=Path,default=DATASET);r.add_argument('--index',type=Path,default=ROOT/'data/index');r.add_argument('--device',default='cuda',choices=['auto','cpu','cuda']);r.add_argument('--top-k',type=int,default=5);r.add_argument('--warmup',type=int,default=1);r.add_argument('--output',type=Path)
 s=sp.add_parser('score');s.add_argument('directory',type=Path);s.add_argument('--review',type=Path)
 a=p.parse_args()
 try:return run(a) if a.cmd=='run' else score(a)
 except (ValueError,OSError,RuntimeError,KeyError,TypeError) as e:print('错误：'+str(e));return 2
if __name__=='__main__':raise SystemExit(main())
