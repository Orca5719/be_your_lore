"""5B experiment identity and offline result presentation."""
import json
import platform
import sys
from pathlib import Path

from .runner import ARMS, read, jsonl, ensure_manifest, validate_arm_identity, validate_logs
from .identity import build_identity, sha
from .model import snapshot_path, versions
from .resources import check_resources

ROOT = Path(__file__).resolve().parents[1]
SCREEN_IDS = ('SL-P03','SL-P04','SL-017','SL-022')
ASSETS = Path('C:/Users/Xhang/Desktop/project/worldcheck')
SOURCE = ASSETS / '.worktrees/benchmark-4.2'


def prepare(source_root, asset_root, phase, order='forward'):
    source_root, asset_root = Path(source_root).resolve(), Path(asset_root).resolve()
    if phase not in ('screen','full') or order not in ('forward','reverse'):
        raise ValueError('invalid phase/order')
    dataset = source_root/'evaluation/story_benchmark_24_v2.json'
    frozen_manifest = source_root/'agent_pipeline_v4_2/reports/benchmark_4_2B_20261017T033125Z_295aa1/manifest.json'
    frozen = read(frozen_manifest)
    paths = [dataset, frozen_manifest]
    if sha(dataset) != frozen['source']['files']['evaluation/story_benchmark_24_v2.json']:
        raise ValueError('dataset differs from reviewed frozen dataset')
    for rel in ('agent_pipeline_v2/extractor.py','agent_pipeline_v2/prompts/extractor_v1.txt',
                'agent_pipeline_v2_1/extractor.py','agent_pipeline_v4/static.py','qwen_judge.py',
                'agent_pipeline_v3/model.py','agent_pipeline_v2/model.py','agent_pipeline_v2/batch_llm.py'):
        if sha(ROOT/rel) != frozen['source']['files'][rel]:
            raise ValueError('local frozen implementation differs: '+rel)
        paths.append(ROOT/rel)
    snapshot = snapshot_path(asset_root)
    paths += [snapshot/name for name in ('config.json','generation_config.json','tokenizer_config.json')]
    index = asset_root/'data/index'
    current = (index/'CURRENT').read_text(encoding='ascii').strip()
    version = (index/'versions'/current).resolve()
    if not version.is_relative_to((index/'versions').resolve()):
        raise ValueError('invalid index version')
    paths += [index/'CURRENT'] + [p for p in version.rglob('*') if p.is_file()]
    paths += [asset_root/'evaluation/agent_pipeline_v2_1_lore_metadata.json']
    paths += [ROOT/name for name in ('encoder.py','retrieval.py','index_store.py','uv.lock')]
    for module in ('agent_pipeline_v2','agent_pipeline_v2_1','agent_pipeline_v2_2','agent_pipeline_v3','agent_pipeline_v4','agent_pipeline_v4_1','agent_pipeline_v4_2','agent_pipeline_v5'):
        paths += [p for p in (ROOT/module).rglob('*') if p.suffix in ('.py','.txt') and 'reports' not in p.relative_to(ROOT/module).parts and '__pycache__' not in p.parts]
    runtime = versions()
    if runtime['transformers'] != '4.57.6':
        raise ValueError('use the existing transformers==4.57.6 environment')
    cases = read(dataset)['cases']
    if len(cases)!=24 or len({c['id'] for c in cases})!=24:
        raise ValueError('expected the frozen 24 distinct stories')
    if phase=='screen':
        by_id={c['id']:c for c in cases}
        cases=[by_id[cid] for cid in SCREEN_IDS]
    # Gold and review never enter the inference cases.
    inputs = [{'id':c['id'],'story':c['story']} for c in cases]
    config = dict(phase=phase,order=order,source_root=str(source_root),asset_root=str(asset_root),
                  arms=list(ARMS if order=='forward' else reversed(ARMS)),runtime=runtime,
                  batch_size=1,prompt_lookup_tokens=5,max_new_tokens=1536,
                  model='Qwen/Qwen3-4B-Instruct-2507',revision=snapshot.name,
                  execution='serial stories; frozen StaticBatcher batch=1',
                  python=sys.version,system=platform.platform(),generation='frozen greedy; do_sample=False',
                  output_start_free_gib=5,output_stop_free_gib=2,report_budget_mib=512)
    return inputs, build_identity(config, paths)


def fresh_review(directory, arm, rows):
    path=directory/arm/'review.json'
    if path.exists():
        return
    labels={}
    pending=[]
    from agent_pipeline_v4_1.audit import event_signature
    for row in rows:
        cid=row['case_id'];labels[cid]={}
        for e in row['result']['benchmark_stages']['extraction']['events']:
            labels[cid][e['id']]=dict(signature=event_signature(e),label=None,gold_ids=[],note=None)
            pending.append(cid+'/'+e['id'])
    from agent_pipeline_v2_2.runner import atomic_write_json
    atomic_write_json(path,dict(schema_version='benchmark-5B-review-v1',arm=arm,cases=labels,pending=pending,
                               extraction_sha256=sha(directory/arm/'extractor_runs.jsonl')))


def summarize(directory):
    from collections import Counter
    from agent_pipeline_v2_2.runner import atomic_write_json
    directory=Path(directory)
    manifest=read(directory/'manifest.json')
    arms={}
    for arm in manifest['config']['arms']:
        output=directory/arm
        if not (output/'extractor_runs.jsonl').exists():
            if (output/'complete.json').exists():
                raise ValueError('completed arm extraction file missing')
            continue
        rows=jsonl(output/'extractor_runs.jsonl');calls=jsonl(output/'inference_calls.jsonl')
        validate_arm_identity(directory,arm,rows)
        validate_logs([r for r in rows if r.get('status')=='ok'],calls,jsonl(output/'wire_calls.jsonl'))
        complete=read(output/'complete.json') if (output/'complete.json').exists() else {}
        if complete:
            for name,digest in complete['files'].items():
                if sha(output/name)!=digest:
                    raise ValueError('arm result changed: '+arm+'/'+name)
        for row in rows:
            if row['case_id'] not in manifest['config']['case_ids']:
                raise ValueError('unknown story in result')
        status=Counter(r['status'] for r in rows)
        count=len(manifest['config']['case_ids'])
        good=len(rows)==count and len({r['case_id'] for r in rows})==count and status['ok']==count and bool(complete)
        if complete and (not good or not complete.get('complete')):
            raise ValueError('completed arm is not complete: '+arm)
        if good:
            fresh_review(directory,arm,rows)
        def total(field):
            values=[r.get(field) for r in calls]
            return sum(values) if all(isinstance(v,(int,float)) for v in values) else None
        review=read(output/'review.json') if (output/'review.json').exists() else None
        if review and review.get('arm')!=arm:
            raise ValueError('review arm identity differs')
        arms[arm]=dict(complete=good,stories=len(rows),status_counts=dict(status),calls=len(calls),
                       wall_seconds=complete.get('wall_seconds') if good else None,
                       input_tokens=total('input_tokens'),output_tokens=total('output_tokens'),
                       prefill_seconds=total('prefill_time'),decode_seconds=total('decode_time'),
                       pending=len(review['pending']) if review else None,**_extra_metrics(rows,calls))
    result=dict(phase=manifest['config']['phase'],arms=arms,
                notice='screen is not formal quality; tokens from real model output before expansion; resumed wall unknown')
    atomic_write_json(directory/'summary.json',result)
    lines=['# Benchmark 5B 四组对照','','短协议只省略空字段及由modality确定的explicit；逐篇串行、batch=1。',
           'Wall包括本组所有正式尝试与适配/检查点写入，模型加载和预热单列。筛选不发布正式质量。','',
           '|组别|完整|故事|调用|输入/输出tokens|Wall秒|待审核|','|---|---|---:|---:|---:|---:|---:|']
    for arm,a in arms.items():
        wall=f"{a['wall_seconds']:.2f}" if a['wall_seconds'] is not None else '—'
        lines.append(f"|{arm}|{a['complete']}|{a['stories']}|{a['calls']}|{a['input_tokens']}/{a['output_tokens']}|{wall}|{a['pending']}|")
    lines += ['','筛选先看状态、tokens、Wall和原始输出，不把pending当质量通过；完整实验需审核review.json，再跑冻结下游。',
              '不同生成模式分别对照；短协议与推测解码收益不可相乘。']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SUMMARY='+str((directory/'summary.md').resolve()),flush=True)
    return result


def run_experiment(args):
    from datetime import datetime
    from .model import load_model
    from .runner import run_extraction_arm
    inputs,identity=prepare(args.source_root,args.asset_root,args.phase,args.order)
    identity['config']['case_ids']=[c['id'] for c in inputs]
    directory=Path(args.output or ROOT/'agent_pipeline_v5/reports'/('benchmark_5B_'+datetime.now().strftime('%Y%m%dT%H%M%S'))).resolve()
    for forbidden in (Path(args.source_root).resolve(),Path(args.asset_root).resolve()/'.cache',Path(args.asset_root).resolve()/'data'):
        if directory.is_relative_to(forbidden) or forbidden.is_relative_to(directory):
            raise ValueError('output overlaps historical source or shared assets')
    check_resources(directory,initial=True)
    ensure_manifest(directory,identity)
    print('RESULT_DIR='+str(directory),flush=True)
    # Validate completed/failed prior arms before loading the GPU model.
    for arm in identity['config']['arms']:
        path=directory/arm
        if (path/'failure.json').exists():
            raise ValueError('prior failed arm preserved; use new result directory')
    summarize(directory)  # Validate any existing arm/log bindings before model load.
    model=None
    for arm in identity['config']['arms']:
        path=directory/arm
        if (path/'complete.json').exists():
            summarize(directory)
            continue
        if model is None:
            model=load_model(args.asset_root)
            from .model import record_setup
            record_setup(model,directory)
        measured=run_extraction_arm(dict(arm=arm,warmup=True,report_root=str(directory)),inputs,model,path)
        summarize(directory)
        if not measured['complete']:
            return 2
    return 0





def _extra_metrics(rows,calls):
    def percentile(values,p):
        if not values or any(v is None for v in values):
            return None
        values=sorted(values);position=(len(values)-1)*p
        lo=int(position);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    ttft=[c.get('ttft_ms') for c in calls]
    stories=[r.get('stage_metrics',{}).get('extraction',{}).get('seconds') for r in rows if r['status']=='ok']
    peaks=[c.get('peak_allocated') for c in calls]
    reserved=[c.get('peak_reserved') for c in calls]
    attempts=[a for r in rows for w in r['result']['benchmark_stages']['extraction'].get('calls',[]) for a in w['attempts']]
    return dict(ttft_p50_ms=percentile(ttft,.5),ttft_p95_ms=percentile(ttft,.95),
                story_p50_seconds=percentile(stories,.5),story_p95_seconds=percentile(stories,.95),
                peak_allocated_bytes=max(peaks) if peaks and all(x is not None for x in peaks) else None,
                peak_reserved_bytes=max(reserved) if reserved and all(x is not None for x in reserved) else None,
                retry_requests=sum(a.get('attempt',1)>1 and a.get('purpose')!='recover_missing_targets' for a in attempts),
                coverage_requests=sum(a.get('purpose')=='recover_missing_targets' for a in attempts))
