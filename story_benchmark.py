"""Run/validate/score the draft Story-Level benchmark without changing canon."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from collections import Counter
from story_benchmark_metrics import score_results,performance_metrics,detection_markdown,unique

ROOT=Path(__file__).resolve().parent
DEFAULT_DATASET=ROOT/'evaluation/story_benchmark_pilot_4_v1.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def validate_dataset(dataset,root,index_directory):
    from index_store import load_index
    cases=dataset.get('cases')
    if not isinstance(cases,list) or not cases:raise ValueError('故事测试集不能为空')
    unique(cases,'id')
    snapshot=dataset['corpus_snapshot']
    actual={p.relative_to(root/'lore').as_posix():digest(p) for p in (root/'lore').rglob('*') if p.suffix.lower() in ('.md','.txt') and p.is_file()}
    if snapshot['files_sha256']!=actual:raise ValueError('资料摘要不匹配，拒绝混用不同世界观基线')
    if (Path(index_directory)/'CURRENT').read_text().strip()!=snapshot['index_version']:
        raise ValueError('索引版本与测试集快照不一致')
    _,metadata=load_index(index_directory)
    chunks=unique(metadata['chunks'],'id')
    if len(chunks)!=snapshot['chunk_count']:raise ValueError('索引片段数不匹配')
    facts=0
    for case in cases:
        story=case['story']
        if not isinstance(story,str) or not 200<=len(story)<=800:raise ValueError('故事必须为200–800字符：'+case['id'])
        if case['character_count']!=len(story):raise ValueError('故事字符数记录错误')
        unique(case['gold_facts'],'id');facts+=len(case['gold_facts'])
        conflicts=[]
        for fact in case['gold_facts']:
            if fact['expected_verdict'] not in ('一致','矛盾','不确定'):raise ValueError('金标判断无效')
            if fact['expected_verdict']=='矛盾':conflicts.append(fact['id'])
            for anchor in [fact['source_anchor']]+fact['context_anchors']:
                start,end=anchor['start'],anchor['end']
                if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(story)
                    or story[start:end]!=anchor['text']):raise ValueError('金标原文位置不一致：'+case['id'])
            evidence=unique(fact['canonical_evidence'],'id')
            for cid,row in evidence.items():
                if cid not in chunks or any(row[key]!=chunks[cid][key] for key in ('text','file','start_line','end_line','heading_path')):
                    raise ValueError('金标设定证据与索引不匹配')
                if row['file'] not in actual:raise ValueError('证据文件不在已冻结资料中')
                lines=(root/'lore'/row['file']).read_text(encoding='utf-8-sig').splitlines()
                if '\n'.join(lines[row['start_line']-1:row['end_line']])!=row['text']:raise ValueError('证据来源行不匹配')
            groups=fact['acceptable_evidence_sets']
            if not isinstance(groups,list) or any(not isinstance(group,list) or not group or any(cid not in evidence for cid in group) for group in groups):
                raise ValueError('可接受证据组合无效')
            if fact['expected_verdict']!='不确定' and not groups:raise ValueError('一致/矛盾金标缺少证据')
        if case['gold_conflict_ids']!=conflicts:raise ValueError('金标冲突清单不一致')
        count=len(conflicts)
        if case['group'] not in ('zero_conflict','one_conflict','multi_conflict') or (case['group']=='zero_conflict' and count!=0) or (case['group']=='one_conflict' and count!=1) or (case['group']=='multi_conflict' and count not in (2,3)):
            raise ValueError('案例分组与矛盾数量不一致')
        expected='矛盾' if conflicts else '不确定' if any(f['expected_verdict']=='不确定' for f in case['gold_facts']) else '一致' if case['gold_facts'] else None
        if case['expected_summary']!=expected:raise ValueError('金标汇总与事实判断不一致')
        for ignored in case['ignored_examples']:
            anchor=ignored['anchor']
            if story[anchor['start']:anchor['end']]!=anchor['text']:raise ValueError('忽略示例位置错误')
    return dict(cases=len(cases),gold_facts=facts,index_chunks=len(chunks),dataset_status=dataset['status'],groups=dict(Counter(case['group'] for case in cases)))


def source_snapshot():
    names=['story_extraction.py','story_retrieval.py','story_judgement.py','story_benchmark.py','story_benchmark_metrics.py',
           'encoder.py','qwen_judge.py','retrieval.py','index_store.py','model_metrics.py',
           'prompts/story_extraction_v2.txt','prompts/story_importance_v2.txt','prompts/story_consistency_v2.txt']
    return {name:digest(ROOT/name) for name in names}


def failed_result(exc):
    return dict(status='error',error=str(exc),extraction=dict(status='error',facts=[],rejected=[],error=str(exc),raw_output=getattr(exc,'raw_output','')),items=[])


def run_cases(cases,process,path,before=None,after=None):
    rows=[]
    with Path(path).open('x',encoding='utf-8') as file:
        for case in cases:
            if before:before()
            started=time.perf_counter();interrupted=False
            try:result=process(case)
            except KeyboardInterrupt:
                result=failed_result(RuntimeError('用户中断当前故事检查'));interrupted=True
            except (ValueError,OSError,RuntimeError) as exc:result=failed_result(exc)
            extras={}
            if after:
                try:extras=after()
                except RuntimeError as exc:extras={'measurement_error':str(exc)}
            generations=result.get('extraction',{}).get('timing',{}).get('calls',[])+[dict(stage='consistency_judgement',**g) for g in result.get('judgement_calls',[])]
            row=dict(case_id=case['id'],seconds=time.perf_counter()-started,result=result,generations=generations,**extras)
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
            rows.append(row)
            print(f"{len(rows)}/{len(cases)} {case['id']} status={result['status']} {row['seconds']:.2f}s",file=sys.stderr,flush=True)
            if interrupted:break
    return rows


def rows_from(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def review_packet(dataset,rows):
    by_case={row['case_id']:row for row in rows}
    lines=['# Story-Level语义配对审核','', '请审阅review.json：每案填写matches并设reviewed=true。没有配对的预测算误提，没有配对的金标算漏提。不要按字面相同或Qwen判断标签配对。','']
    for case in dataset['cases']:
        result=by_case.get(case['id'],{}).get('result',{})
        lines+=['## '+case['id']+' '+case.get('title',''),'',case['story'],'','金标：']
        for gold in case['gold_facts']:
            lines.append('- '+gold['id']+' '+gold['subject']+' → '+gold['predicate']+' → '+str(gold['object'])+'；预期 '+gold['expected_verdict']+'；原文 '+gold['source_anchor']['text'])
        lines+=['','预测（原始证据、上下文及失败详情见raw.jsonl）：']
        for fact in result.get('extraction',{}).get('facts',[]):
            lines+=['- '+fact['id']+' '+fact['subject']+' → '+fact['predicate']+' → '+str(fact.get('object'))+'；原文 '+fact['source_text'],'  必要上下文：'+fact.get('context_text','')]
        lines+=['','处理状态：'+result.get('status','未运行'),'']
    return '\n'.join(lines)


def performance_markdown(perf,report):
    from model_metrics import baseline_markdown
    display=dict(perf)
    # Always show measured request latency, without mislabelling failed warm-up as warm.
    display['warm_total_latency_median_seconds']=perf['total_latency_median_seconds']
    text=baseline_markdown(display).replace('Total latency (warm request median)','Total latency (measured story median)')
    def number(value,unit):return f'{value:.3f} {unit}' if value is not None else 'N/A'
    text+='\n| Story performance | Value |\n|---|---|\n'
    for name,value in [('Model load',number(report.get('model_load_seconds'),'s')),
                       ('BGE/index load',number(report.get('encoder_index_load_seconds'),'s')),
                       ('Story latency P95',number(perf['total_latency_p95_seconds'],'s')),
                       ('Sequential story throughput',number(perf['story_throughput_per_second'],'stories/s')),
                       ('Measured LLM calls (including retries)',str(perf['generation_calls'])),
                       ('Warm-up attempts / failures',str(report.get('warmup_count',0))+' / '+str(report.get('warmup_failed',0)))]:
        text+='| '+name+' | '+value+' |\n'
    text+='\nStage status/counters: `'+json.dumps(report.get('stage_summary',{}),ensure_ascii=False)+'`\n'
    return text


def run(args):
    dataset=read_json(args.dataset)
    validation=validate_dataset(dataset,ROOT,args.index)
    if args.warmup<0 or args.top_k<1:raise ValueError('warmup不可负，top-k必须为正数')
    directory=args.output or ROOT/'reports'/('story_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'dataset.json').write_bytes(args.dataset.read_bytes())
    report=dict(name='Story-Level Consistency Benchmark',run_id=directory.name,dataset_sha256=digest(directory/'dataset.json'),
                device_requested=args.device,top_k=args.top_k,warmup_count=args.warmup,validation=validation,source_sha256=source_snapshot(),status='loading')
    write_json(directory/'report.json',report)
    rows=[];warmups=[]
    try:
        from qwen_judge import QwenJudge,MODEL,REVISION
        from encoder import Encoder
        from retrieval import Retriever
        from story_judgement import check_story
        judge=QwenJudge(args.device)
        started=time.perf_counter()
        encoder=Encoder(device=judge.device)
        retriever=Retriever(args.index,encoder)
        report.update(model=MODEL,model_revision=REVISION,device=judge.device,model_load_seconds=judge.load_seconds,
            encoder_index_load_seconds=time.perf_counter()-started,encoder_config=encoder.config,index_config=retriever.metadata['config'],
            model_footprint_bytes=judge.model_footprint_bytes,model_cuda_allocated_bytes=judge.model_cuda_allocated_bytes)
        import platform
        from importlib.metadata import version,PackageNotFoundError
        packages={}
        for name in ('torch','transformers','numpy','bitsandbytes','huggingface_hub'):
            try:packages[name]=version(name)
            except PackageNotFoundError:packages[name]=None
        report['environment']=dict(python=platform.python_version(),platform=platform.platform(),
            packages=packages,
            torch_cpu_threads=judge.torch.get_num_threads(),gpu=judge.torch.cuda.get_device_name() if judge.device=='cuda' else None,
            gpu_total_bytes=judge.torch.cuda.get_device_properties(0).total_memory if judge.device=='cuda' else None)
        def process(case):return check_story(case['story'],judge.device,args.index,args.top_k,judge=judge,retriever=retriever)
        for index in range(args.warmup):
            print(f'预热 {index+1}/{args.warmup}，不计检测指标。',file=sys.stderr,flush=True)
            try:warmups.append(process({'story':'雷感觉左胸里的亚巴顿开始躁动。'}))
            except (ValueError,OSError,RuntimeError) as exc:warmups.append(failed_result(exc))
        write_json(directory/'warmup.json',warmups)
        def before():
            if judge.device=='cuda':
                judge.torch.cuda.synchronize();judge.torch.cuda.reset_peak_memory_stats()
        def after():
            if judge.device!='cuda':return {}
            judge.torch.cuda.synchronize()
            return dict(peak_allocated_bytes=judge.torch.cuda.max_memory_allocated(),peak_reserved_bytes=judge.torch.cuda.max_memory_reserved())
        rows=run_cases(dataset['cases'],process,directory/'raw.jsonl',before,after)
        report['status']='completed' if len(rows)==len(dataset['cases']) else 'interrupted'
    except (ValueError,OSError,RuntimeError,KeyboardInterrupt) as exc:
        report.update(status='error',error=str(exc) or '用户中断')
    finally:
        if (directory/'raw.jsonl').exists():rows=rows_from(directory/'raw.jsonl')
        else:(directory/'raw.jsonl').write_text('',encoding='utf-8')
        report.update(saved_cases=len(rows),planned_cases=len(dataset['cases']),raw_sha256=digest(directory/'raw.jsonl'),scoring_status='pending_semantic_review')
        try:
            validate_dataset(dataset,ROOT,args.index)
            if source_snapshot()!=report['source_sha256']:raise ValueError('运行过程中代码或提示摘要发生变化')
            report['snapshot_stable']=True
        except (ValueError,OSError,KeyError) as exc:report.update(snapshot_stable=False,snapshot_error=str(exc))
        for key in ('peak_allocated_bytes','peak_reserved_bytes'):
            values=[r[key] for r in rows if r.get(key) is not None]
            report[key]=max(values) if values else None
        report['warmup_failed']=sum(r.get('status')!='ok' for r in warmups)
        all_items=[item for row in rows for item in row['result'].get('items',[])]
        report['stage_summary']=dict(
            case_status_counts=dict(Counter(row['result']['status'] for row in rows)),
            extraction_status_counts=dict(Counter(row['result'].get('extraction',{}).get('status','missing') for row in rows)),
            retrieval_failed_facts=sum(x.get('status')!='ok' for x in all_items),
            judgement_failed_facts=sum(x.get('judgement',{}).get('status')!='ok' for x in all_items),
            pending_candidates=sum(len(row['result'].get('extraction',{}).get('pending_facts',[])) for row in rows),
            rejected_candidates=sum(len(row['result'].get('extraction',{}).get('rejected',[])) for row in rows),
            retries=sum(max(0,len(x.get('judgement',{}).get('attempts',[]))-1) for x in all_items),
            normalized_replies=sum(bool(x.get('judgement',{}).get('format_normalizations')) for x in all_items))
        write_json(directory/'report.json',report)
        review=dict(run_id=report['run_id'],dataset_sha256=report['dataset_sha256'],raw_sha256=report['raw_sha256'],
                    cases=[dict(case_id=case['id'],reviewed=False,matches=[],notes='') for case in dataset['cases']])
        write_json(directory/'review.json',review)
        (directory/'review.md').write_text(review_packet(dataset,rows),encoding='utf-8')
        if 'model' in report:
            perf=performance_metrics(report,rows)
            write_json(directory/'performance.json',perf)
            (directory/'benchmark.md').write_text('# Story-Level Consistency Benchmark — pilot/draft\n\n检测指标待语义配对审核，不能只按字符串或模型标签计分。\n\n'+performance_markdown(perf,report),encoding='utf-8')
        print('结果目录：'+str(directory),flush=True)
    return 0 if report['status']=='completed' and report['snapshot_stable'] else 2


def score(args):
    directory=args.run_directory
    report=read_json(directory/'report.json')
    review=read_json(args.review or directory/'review.json')
    if not report.get('snapshot_stable'):raise ValueError('运行时快照不稳定，不能计分')
    for key in ('run_id','dataset_sha256','raw_sha256'):
        if review.get(key)!=report[key]:raise ValueError('审核记录不属于本次运行：'+key)
    if digest(directory/'dataset.json')!=report['dataset_sha256'] or digest(directory/'raw.jsonl')!=report['raw_sha256']:
        raise ValueError('运行资料或原始结果摘要不一致，拒绝计分')
    dataset=read_json(directory/'dataset.json');rows=rows_from(directory/'raw.jsonl')
    metrics=score_results(dataset,rows,review,k=report['top_k'])
    metrics.update(run_id=report['run_id'],dataset_sha256=report['dataset_sha256'],raw_sha256=report['raw_sha256'],
                   review_sha256=digest(args.review or directory/'review.json'),scorer_sha256=digest(ROOT/'story_benchmark_metrics.py'))
    write_json(directory/'metrics.json',metrics)
    perf=performance_metrics(report,rows) if 'model' in report else None
    text='# Story-Level Consistency Benchmark — pilot/draft\n\n'+detection_markdown(metrics)
    if perf:
        perf['accuracy']=metrics['classification']['accuracy']
        perf['measurement']=perf['measurement'].replace('Accuracy pending reviewed semantic matches.','Accuracy from reviewed semantic matches; all gold facts included.')
        text+='\n'+performance_markdown(perf,report)
    (directory/'benchmark.md').write_text(text,encoding='utf-8')
    print(text)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('validate','run'):
        p=sub.add_parser(command);p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET);p.add_argument('--index',type=Path,default=ROOT/'data/index')
        if command=='run':
            p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--warmup',type=int,default=1)
            p.add_argument('--top-k',type=int,default=5);p.add_argument('--output',type=Path)
    p=sub.add_parser('score');p.add_argument('run_directory',type=Path);p.add_argument('--review',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':return run(args)
        if args.command=='score':return score(args)
        print(json.dumps(validate_dataset(read_json(args.dataset),ROOT,args.index),ensure_ascii=False));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print('错误：'+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
