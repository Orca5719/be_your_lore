"""160-case frozen application benchmark. Never changes production prompts or lore."""
import argparse
from collections import Counter
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time

ROOT=Path(__file__).resolve().parent
SCORING_VERSION='world-benchmark-multiview-v2'
LABELS=('一致','矛盾','不确定','缺失/失败')
DIMENSIONS=('时间','人物知识','空间','人物关系','物理规则','世界规则','因果','身份')
FIXTURES=('characters.md','angels.md','history.md','technology.md')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def aggregate(verdicts):
    if not verdicts or any(v not in LABELS[:3] for v in verdicts):return '缺失/失败'
    return '矛盾' if '矛盾' in verdicts else '不确定' if '不确定' in verdicts else '一致'

def score_case(case,preview):
    items=preview.get('items',[])
    facts=[i for i in items if i['proposed']['kind'] not in ('entity','timepoint')]
    observed=aggregate([i['review'].get('verdict') if i['review']['status']=='ok' else None for i in facts]) if preview.get('status')=='pending_confirmation' else '缺失/失败'
    score=dict(claims_total=len(case['claims']),claims_found=0,verdict_correct=0,kind_correct=0,subject_total=0,subject_correct=0,context_total=0,context_correct=0,structures_total=len(case.get('required_structures',[])),structures_found=0,citations_total=0,citations_invalid=0,retrieval_total=0,retrieval_hits=0,citation_core_hits=0,unmatched_facts=0,details=[],observed_overall=observed,overall_correct=observed==case['expected_overall'])
    sources={r['id']:r['text'] for r in preview.get('judgment',{}).get('evidence',[])}
    for item in items:
        for citation in item['review'].get('evidence',[]):
            score['citations_total']+=1
            if citation.get('chunk_id') not in sources or not citation.get('quote') or citation['quote'] not in sources[citation['chunk_id']]:score['citations_invalid']+=1
    matched=set()
    for gold in case['claims']:
        candidates=[(n,i) for n,i in enumerate(facts) if gold['input_quote'] in i['proposed']['text']]
        found=bool(candidates) and len({i['proposed']['text'] for _,i in candidates})==1
        item=candidates[0][1] if found else None
        if found:matched.update(n for n,_ in candidates)
        verdict=aggregate([i['review'].get('verdict') if i['review']['status']=='ok' else None for _,i in candidates])
        verdict_ok=found and all(i['review']['status']=='ok' and i['review'].get('verdict')==gold['expected_verdict'] for _,i in candidates)
        kind_ok=found and all(i['proposed']['kind'] in gold['allowed_kinds'] for _,i in candidates)
        subjects=gold.get('subject_any',[]);terms=gold.get('context_terms',[])
        subject_ok=found and all(i['proposed'].get('entity') in subjects for _,i in candidates) if subjects else True
        context=' '.join(v for v in item['proposed'].values() if isinstance(v,str)) if found else ''
        context_ok=all(t in context for t in terms) if found else not terms
        score['claims_found']+=found;score['verdict_correct']+=verdict_ok;score['kind_correct']+=kind_ok
        if subjects:score['subject_total']+=1;score['subject_correct']+=subject_ok
        if terms:score['context_total']+=1;score['context_correct']+=context_ok
        # Unknown claims may carry contextual evidence, but it does not establish the novel claim.
        core=[e['quote'] for e in gold['evidence']] if gold['expected_verdict'] in LABELS[:2] else []
        retrieved=[]
        if found:
            retrieved=[r for check in preview.get('checks',[]) if check['text']==item['proposed']['text'] for r in check['result'].get('evidence',[])]
        retrieval_hit=any(q in r['text'] for q in core for r in retrieved)
        citation_hit=found and any(q in c.get('quote','') for q in core for c in item['review'].get('evidence',[]))
        if core:score['retrieval_total']+=1;score['retrieval_hits']+=retrieval_hit;score['citation_core_hits']+=citation_hit
        score['details'].append(dict(input_quote=gold['input_quote'],expected=gold['expected_verdict'],predicted=verdict,found=found,kind=item['proposed']['kind'] if found else None,kind_correct=bool(kind_ok),subject_correct=bool(subject_ok),context_correct=bool(context_ok),retrieval_core_hit=bool(retrieval_hit),citation_core_hit=bool(citation_hit)))
    score['unmatched_facts']=len(facts)-len(matched)
    score['structures_found']=sum(any(all(i['proposed'].get(k)==v for k,v in s.items()) for i in items) for s in case.get('required_structures',[]))
    score['complete']=bool(preview.get('status')=='pending_confirmation' and score['overall_correct'] and score['verdict_correct']==score['claims_total'] and score['kind_correct']==score['claims_total'] and score['subject_correct']==score['subject_total'] and score['context_correct']==score['context_total'] and score['structures_found']==score['structures_total'] and score['citations_invalid']==0 and score['unmatched_facts']==0)
    return score

def validate_dataset(data,lore):
    from world_extraction import source_spans
    cases=data['cases']
    if len(cases)!=160 or len({c['id'] for c in cases})!=160 or len({c['text'] for c in cases})!=160:raise ValueError('必须有160条唯一输入与ID')
    if Counter(c['group'] for c in cases)!=dict(consistent=40,contradiction=80,insufficient=20,unrelated=20):raise ValueError('组别分布不匹配')
    if Counter(c['primary_dimension'] for c in cases if c['group']=='contradiction')!={d:10 for d in DIMENSIONS}:raise ValueError('八类矛盾必须各10条')
    for f,h in data['fixture_hashes'].items():
        if f not in FIXTURES or sha(lore/f)!=h:raise ValueError('资料摘要不匹配：'+f)
    for case in cases:
        if not case['claims'] or aggregate([c['expected_verdict'] for c in case['claims']])!=case['expected_overall']:raise ValueError('整体与逐事实标注不一致：'+case['id'])
        for c in case['claims']:
            if not any(c['input_quote'] in span for span in source_spans(case['text']).values()):raise ValueError('事实片段跨越分句，请将前置条件另行标注：'+case['id'])
            if c['input_quote'] not in case['text'] or not c['input_quote'] or not c['rationale']:raise ValueError('标注片段/理由无效：'+case['id'])
            if c['expected_verdict'] in LABELS[:2] and not c['evidence']:raise ValueError('一致/矛盾必须有证据：'+case['id'])
            if any(t not in case['text'] for t in c.get('context_terms',[])):raise ValueError('上下文标注不在输入中')
            if c.get('subject_any') and not any(s in case['text'] for s in c['subject_any']):raise ValueError('主体标注不在输入中')
            for e in c['evidence']:
                if e['file'] not in FIXTURES:raise ValueError('来源越界')
                lines=(lore/e['file']).read_text(encoding='utf-8-sig').splitlines()
                if not 1<=e['start_line']<=e['end_line']<=len(lines) or e['quote'] not in '\n'.join(lines[e['start_line']-1:e['end_line']]):raise ValueError('引文定位错误：'+case['id'])
        for s in case.get('required_structures',[]):
            if s['entity'] not in case['text']:raise ValueError('结构名称不在输入中')
    return dict(cases=len(cases),claims=sum(len(c['claims']) for c in cases),mixed=sum(len(c['claims'])>1 for c in cases),families=len({c['family_id'] for c in cases}),evidence_families=len({f for c in cases for f in c['evidence_families']}))

def totals(rows):
    counters=Counter()
    for row in rows:
        counters.update({k:int(v) for k,v in row['score'].items() if isinstance(v,(bool,int))})
    counters['inputs']=len(rows);counters['valid_previews']=sum(r['code']==0 for r in rows)
    return dict(counters)

def latency(values):
    if not values:return dict(n=0,median=None,p95=None)
    import numpy as np
    return dict(n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),min=min(values),max=max(values))

def markdown(report):
    t=report['totals'];lines=['# 160条世界观benchmark v1','',report['annotation_policy'],'',f"配置：{report['model']} @ {report['revision']}，{report['device']}；Qwen NF4/BF16，BGE CPU FP32，Top5。当前生产实现与提示冻结，不重试、不边测边调。",'',f"- 完成输入：{t['inputs']}/160；有效预览：{t['valid_previews']}/160",f"- 整体标签符合预期：{t['overall_correct']}/160（单独指标，不代表全部事实覆盖）",f"- 核心事实覆盖：{t['claims_found']}/{t['claims_total']}",f"- 逐事实判定正确：{t['verdict_correct']}/{t['claims_total']}，遗漏/失败在分母",f"- 允许kind符合：{t['kind_correct']}/{t['claims_total']}",f"- 显式标注的主体字段：{t['subject_correct']}/{t['subject_total']}；上下文关键字保留：{t['context_correct']}/{t['context_total']}",f"- 期望新增结构：{t['structures_found']}/{t['structures_total']}",f"- 所有自动标准均满足：{t['complete']}/160",f"- 未匹配到人工事实的额外提取：{t['unmatched_facts']}（不自动断言这些全是幻觉，需人工复核）",f"- 支持/矛盾核心证据Top5：{t['retrieval_hits']}/{t['retrieval_total']}，包含因未提取而未检索",f"- 输出引用核心证据：{t['citation_core_hits']}/{t['retrieval_total']}",f"- 无效来源/原文引用：{t['citations_invalid']}/{t['citations_total']}",'','## 按输入类型','', '| 类型 | 输入数 | 有效预览 | 整体标签正确 | 事实判定正确 | 全项满足 |','|---|---:|---:|---:|---:|---:|']
    for group,s in report['by_group'].items():lines.append(f"| {group} | {s['inputs']} | {s['valid_previews']} | {s['overall_correct']} | {s['verdict_correct']}/{s['claims_total']} | {s['complete']} |")
    lines+=['','## 八类矛盾','', '| 主要维度 | 整体判定正确/输入 | 事实判定正确 |','|---|---:|---:|']
    for dim,s in report['by_conflict_dimension'].items():lines.append(f"| {dim} | {s['overall_correct']}/{s['inputs']} | {s['verdict_correct']}/{s['claims_total']} |")
    lines+=['','## 逐事实混淆矩阵','', '| 预期\\输出 | 一致 | 矛盾 | 不确定 | 缺失/失败 |','|---|---:|---:|---:|---:|']
    for label in LABELS[:3]:lines.append('| '+label+' | '+' | '.join(str(report['confusion'][label][o]) for o in LABELS)+' |')
    lines+=['','## 耗时与边界','',f"- 首例{report['first_request_seconds']:.2f}秒，含Qwen加载{report['qwen_load_seconds']:.2f}秒；编码器/隔离索引已准备，不是完整新进程冷启动。"]
    for label,key in [('后续所有请求，含失败','warm_all'),('后续API有效预览','warm_valid'),('后续全项满足输入','warm_complete')]:
        s=report[key];lines.append(f"- {label}：n={s['n']}，中位={s['median']}秒，P95={s['p95']}秒。")
    lines += [f"- 生成总吞吐{report['generation_throughput']['tokens_per_second']:.2f} tokens/秒；峰值已分配GPU显存{report['peak_allocated_mib']:.1f}MiB，不含所有驱动/其他进程显存。",'- 单次运行，无重复测量；CPU LLM性能本次未测。',f"- 正式资料/索引文件未变：{report['original_data_unchanged']}；冻结摘要未变：{report['frozen_manifest_unchanged']}。",f"- {report['dataset_counts']['families']}个family，{report['dataset_counts']['evidence_families']}个引文来源组；不同措辞并非独立新世界。标注由实现者自查，没有独立专家复核。",'- 上下文只检查显式标注的文字是否保留，不证明时间/条件推理正确。kind只评允许集合，不全面评实体类型和细分类别。引用有效不保证推理原因合理。','- 原始输入、提取与判断输出、证据、逐例耗时/tokens/显存均见cases.jsonl；manifest记录实现、提示、依赖锁和数据摘要。','', '## 模型输出错误类型','']
    for error,n in report['errors'].items():lines.append(f'- {n}例：{error}')
    lines+=['','## 逐例','', '| ID | 组别 | 维度 | 预期 | 实际整体 | 事实正确 | 秒 | 全项满足 |','|---|---|---|---|---|---:|---:|---|']
    for row in report['case_summary']:
        s=row['score'];lines.append(f"| {row['id']} | {row['group']} | {row['primary_dimension']} | {row['expected_overall']} | {s['observed_overall']} | {s['verdict_correct']}/{s['claims_total']} | {row['seconds']:.2f} | {'是' if s['complete'] else '否'} |")
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda');p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    dataset=ROOT/'evaluation/world_benchmark_160_v1.json';data=json.loads(dataset.read_text(encoding='utf-8'));counts=validate_dataset(data,ROOT/'lore')
    if args.validate_only:print(json.dumps(counts,ensure_ascii=False));return
    from encoder import Encoder
    from index_store import build_index
    from qwen_judge import QwenJudge,MODEL,REVISION
    import torch
    import world_entry
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+__import__('uuid').uuid4().hex[:8]
    output=ROOT/'reports/world_benchmark_160_v1'/run_id;output.mkdir(parents=True)
    frozen=[dataset,ROOT/'world_benchmark.py',ROOT/'evaluation/author_world_benchmark_160.py',ROOT/'uv.lock']+[ROOT/name for name in ['model_metrics.py','world_entry.py','entry.py','qwen_judge.py','world_extraction.py','world_records.py','entries.py','encoder.py','documents.py','index_store.py','retrieval.py','judgment_protocol.py','prompts/world_extraction_v4.txt','prompts/world_review_v2.txt']]
    manifest={str(f.relative_to(ROOT)):sha(f) for f in frozen}
    def originals():
        paths=sorted(p for folder in [ROOT/'lore',ROOT/'data/index'] for p in folder.rglob('*') if p.is_file())+[ROOT/'data/entries.json',ROOT/'data/world_records.json']
        return {str(p.relative_to(ROOT)):sha(p) if p.exists() else None for p in paths}
    before=originals()
    report=dict(scoring_version=SCORING_VERSION,status='running',run_id=run_id,created_at=datetime.now(timezone.utc).isoformat(),model=MODEL,revision=REVISION,requested_device=args.device,device=args.device,annotation_policy=data['annotation_policy'],dataset_counts=counts,manifest=manifest,fixture_hashes=data['fixture_hashes'],hardware={'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/2**20 if torch.cuda.is_available() else None},packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','bitsandbytes']},platform=platform.platform(),python=platform.python_version(),completed=0)
    rows=[];generations=[];generate=QwenJudge._generate
    def measured(judge,*a,**kw):
        try:return generate(judge,*a,**kw)
        finally:generations.append(dict(judge.last_generation))
    QwenJudge._generate=measured
    def persist():
        target=output/'summary.json';temp=output/'summary.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(target)
    persist();print('REPORT_DIR='+str(output),flush=True)
    with tempfile.TemporaryDirectory(prefix='worldcheck-benchmark160-') as tmp,(output/'cases.jsonl').open('w',encoding='utf-8') as traces:
        directory=Path(tmp);lore=directory/'lore';lore.mkdir()
        for f in FIXTURES:shutil.copy2(ROOT/'lore'/f,lore/f)
        started=time.perf_counter();encoder=Encoder(device='cpu');report['encoder_load_seconds']=time.perf_counter()-started
        # Validate all queries before starting the timed inference run, without truncation.
        encoder._checked_texts([encoder.config['query_prompt']+c['text'] for c in data['cases']])
        started=time.perf_counter();index=directory/'index';build_index(lore,index,encoder);report['fixture_index_seconds']=time.perf_counter()-started
        session={'encoder':encoder}
        options=['--lore',str(lore),'--index',str(index),'--store',str(directory/'world.json'),'--legacy-store',str(directory/'legacy.json'),'--device',args.device,'--preview-only','--json']
        for case in data['cases']:
            generations.clear()
            if torch.cuda.is_available() and args.device!='cpu':torch.cuda.reset_peak_memory_stats()
            buf=io.StringIO();started=time.perf_counter()
            with contextlib.redirect_stdout(buf):code=world_entry.main([case['text']]+options,session)
            seconds=time.perf_counter()-started;preview=json.loads(buf.getvalue())
            row=dict(**case,code=code,seconds=seconds,preview=preview,score=score_case(case,preview),generations=list(generations),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() and args.device!='cpu' else 0,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_available() and args.device!='cpu' else 0)
            traces.write(json.dumps(row,ensure_ascii=False)+'\n');traces.flush();rows.append(row)
            report['completed']=len(rows);report['progress_totals']=totals(rows);persist()
            print(json.dumps(dict(completed=len(rows),id=case['id'],seconds=round(seconds,2),expected=case['expected_overall'],predicted=row['score']['observed_overall'],facts_correct=row['score']['verdict_correct'],facts_total=row['score']['claims_total']),ensure_ascii=False),flush=True)
        if (directory/'world.json').exists() or (directory/'legacy.json').exists():raise RuntimeError('预览产生了写入')
        report['qwen_load_seconds']=session['judge'].load_seconds;report['device']=session['judge'].device
        report['model_footprint_bytes']=session['judge'].model_footprint_bytes
        report['model_cuda_allocated_bytes']=session['judge'].model_cuda_allocated_bytes
    report['original_data_unchanged']=originals()==before;report['frozen_manifest_unchanged']=all(sha(ROOT/f)==h for f,h in manifest.items())
    report['totals']=totals(rows)
    report['by_group']={g:totals([r for r in rows if r['group']==g]) for g in data['group_targets']}
    report['by_conflict_dimension']={d:totals([r for r in rows if r['group']=='contradiction' and r['primary_dimension']==d]) for d in DIMENSIONS}
    report['family_scores']={f:totals([r for r in rows if r['family_id']==f]) for f in sorted({r['family_id'] for r in rows})}
    report['confusion']={label:{o:sum(d['expected']==label and d['predicted']==o for r in rows for d in r['score']['details']) for o in LABELS} for label in LABELS[:3]}
    report['first_request_seconds']=rows[0]['seconds'];report['warm_all']=latency([r['seconds'] for r in rows[1:]]);report['warm_valid']=latency([r['seconds'] for r in rows[1:] if r['code']==0]);report['warm_complete']=latency([r['seconds'] for r in rows[1:] if r['score']['complete']]);report['peak_allocated_mib']=max(r['peak_allocated_mib'] for r in rows)
    gs=[g for r in rows for g in r['generations'] if g.get('seconds',0)>0];gtime=sum(g['seconds'] for g in gs);gtokens=sum(g['generated_tokens'] for g in gs)
    report['generation_throughput']=dict(seconds=gtime,generated_tokens=gtokens,tokens_per_second=gtokens/gtime if gtime else 0)
    report['peak_reserved_mib']=max(r['peak_reserved_mib'] for r in rows)
    from model_metrics import metric_baseline,baseline_markdown
    report['metric_baseline']=metric_baseline(report,rows)
    (output/'metric_baseline.md').write_text(baseline_markdown(report['metric_baseline']),encoding='utf-8')
    report['errors']=dict(Counter(r['preview'].get('error') or '分项核对失败' for r in rows if r['code']!=0))
    report['case_summary']=[{k:r[k] for k in ['id','group','primary_dimension','expected_overall','seconds','code','score']} for r in rows]
    report['status']='complete' if report['original_data_unchanged'] and report['frozen_manifest_unchanged'] else 'invalidated';persist()
    (output/'report.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],report_dir=str(output),totals=report['totals']),ensure_ascii=False),flush=True)
    if report['status']!='complete':raise RuntimeError('运行期间正式数据或冻结配置改变')

if __name__=='__main__':main()
