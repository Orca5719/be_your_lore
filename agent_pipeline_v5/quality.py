"""Formal quality gates and frozen Hybrid/Judge/Report replay for 5B."""
from pathlib import Path
from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from agent_pipeline_v4_1.audit import validate_labels, score_extraction, score_end_to_end
from .runner import ARMS, read, jsonl, ensure_manifest, validate_arm_identity, validate_logs
from .identity import sha
from .resources import check_resources


def reviewed_arm(directory, arm):
    directory=Path(directory)
    manifest=read(directory/'manifest.json')
    if manifest['config']['phase']!='full':
        raise ValueError('formal scoring requires a full run, not screen')
    if arm not in manifest['config']['arms']:
        raise ValueError('unknown arm')
    config=manifest['config']
    dataset=Path(config['source_root'])/'evaluation/story_benchmark_24_v2.json'
    if manifest['files'].get(str(dataset.resolve()))!=sha(dataset):
        raise ValueError('dataset changed since extraction')
    cases=read(dataset)['cases']
    output=directory/arm
    complete=read(output/'complete.json')
    for name,digest in complete['files'].items():
        if sha(output/name)!=digest:
            raise ValueError('completed extraction changed')
    rows=jsonl(output/'extractor_runs.jsonl')
    validate_arm_identity(directory,arm,rows)
    validate_logs(rows,jsonl(output/'inference_calls.jsonl'),jsonl(output/'wire_calls.jsonl'))
    if not complete.get('complete') or any(r.get('status')!='ok' or r['result']['benchmark_stages']['extraction'].get('status')!='ok' for r in rows):
        raise ValueError('incomplete extraction')
    by_id={c['id']:c for c in cases}
    if len(cases)!=24 or len(by_id)!=24:
        raise ValueError('expected 24 unique full cases')
    for row in rows:
        if row['result']['benchmark_stages']['extraction']['text']!=by_id[row['case_id']]['story']:
            raise ValueError('story text changed')
    review=read(output/'review.json')
    if review.get('arm') != arm:
        raise ValueError('review arm identity differs')
    if review.get('pending') or review.get('extraction_sha256')!=sha(output/'extractor_runs.jsonl'):
        raise ValueError('review pending or extraction digest changed')
    validate_labels(cases,rows,review['cases'])
    return cases,rows,review


def score(directory, arms=None):
    from .diagnostics import validate_arm
    directory=Path(directory); manifest=read(directory/'manifest.json')
    selected=arms or manifest['config']['arms']
    scores={}
    for arm in selected:
        cases,rows,review=reviewed_arm(directory,arm)
        q={'extraction':score_extraction(cases,rows,review['cases']),'end_to_end':None,'complete':False}
        path=directory/arm/'downstream_runs.jsonl'
        if path.exists():
            bound=read(directory/arm/'downstream_manifest.json')
            if bound.get('run_manifest_sha256') != sha(directory/'manifest.json'):
                raise ValueError('downstream run manifest changed')
            if bound.get('arm')!=arm:
                raise ValueError('downstream arm identity changed')
            if bound['review_sha256']!=sha(directory/arm/'review.json') or bound['extraction_sha256']!=sha(directory/arm/'extractor_runs.jsonl'):
                raise ValueError('downstream review or extraction no longer matches')
            down=jsonl(path)
            validate_arm(cases,rows,down,review)
            q['end_to_end']=score_end_to_end(cases,down,review['cases'])
            q['complete']=True
        scores[arm]=q
    atomic_write_json(directory/'quality_reviewed.json',scores)
    name=manifest['config'].get('experiment','5B')
    lines=[f'# Benchmark {name} 审核后质量','','历史gold争议保持原口径；本轮只改变manifest声明的单项策略。','',
           '|组别|提取Recall|提取Precision|幻觉|E2E TP/FP/FN|E2E F1|完整下游|','|---|---:|---:|---:|---:|---:|---|']
    for arm,q in scores.items():
        e=q['extraction'];j=q['end_to_end']
        counts=f"{j['tp']}/{j['fp']}/{j['fn']}" if j else '—'
        f1=f"{j['f1']:.2%}" if j and j['f1'] is not None else '—'
        lines.append(f"|{arm}|{e['recall']:.2%}|{e['precision']:.2%}|{e['hallucinated_events']}|{counts}|{f1}|{q['complete']}|")
    (directory/'quality_reviewed.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('QUALITY='+str((directory/'quality_reviewed.md').resolve()),flush=True)
    return scores


def run_downstream(args):
    from .experiment import prepare
    from .model import load_model, load_retriever
    from .diagnostics import validate_arm
    from agent_pipeline_v4_1.cli import _run_one
    directory=Path(args.result_dir).resolve(); manifest=read(directory/'manifest.json'); config=manifest['config']
    _,current=prepare(config['source_root'],config['asset_root'],config['phase'],config['order'],config.get('experiment','5B'))
    current['config']['case_ids']=config['case_ids']
    ensure_manifest(directory,current)
    selected=args.arms or config['arms']
    # Check all reviews, resource gates, and existing bindings before loading model.
    checked={arm:reviewed_arm(directory,arm) for arm in selected}
    check_resources(directory,initial=True)
    for arm in selected:
        out=directory/arm
        binding={'arm':arm,'run_manifest_sha256':sha(directory/'manifest.json'),
                 'review_sha256':sha(out/'review.json'),'extraction_sha256':sha(out/'extractor_runs.jsonl')}
        path=out/'downstream_manifest.json'
        if path.exists() and read(path)!=binding:
            raise ValueError('downstream identity changed')
        if not path.exists():
            atomic_write_json(path,binding)
        down=jsonl(out/'downstream_runs.jsonl')
        cases,_,_=checked[arm]
        ids=[c['id'] for c in cases]
        if [r['case_id'] for r in down]!=ids[:len(down)] or any(r.get('status')!='ok' for r in down):
            raise ValueError('partial/invalid saved downstream preserved; use a new result directory')
    llm=None;retriever=None
    for arm in selected:
        cases,rows,review=checked[arm]; out=directory/arm
        by_id={r['case_id']:r for r in rows};down=jsonl(out/'downstream_runs.jsonl')
        if len(down)<len(cases) and llm is None:
            llm=load_model(config['asset_root']);llm.enabled=False
            retriever=load_retriever(config['asset_root'],llm.device)
        for case in cases[len(down):]:
            check_resources(directory)
            llm.set_story_id(arm+':'+case['id'])
            ex=by_id[case['id']]['result']['benchmark_stages']['extraction']
            try:
                result=_run_one(ex,llm,retriever)
            except Exception as exc:
                result={'status':'error','error':str(exc)}
            down.append({'case_id':case['id'],'system':arm,**result})
            write_jsonl_atomic(out/'downstream_runs.jsonl',down)
            print(f'[{arm} downstream] {len(down)}/{len(cases)} {case["id"]} {result["status"]}',flush=True)
            llm.clear_traces()
            if result['status']!='ok':
                return 2
        validate_arm(cases,rows,down,review)
    score(directory,selected)
    return 0
