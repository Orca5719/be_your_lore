"""Read-only 5C1 field repair replay, not a model or quality benchmark."""
from pathlib import Path
import agent_pipeline_v2.extractor as validator
from agent_pipeline_v2.extractor import _normalize_window_value, _decode_window
from agent_pipeline_v2_2.runner import atomic_write_json
from .repair import repair_raw, POLICY
from .runner import read,jsonl,validate_arm_identity,validate_logs
from .identity import build_identity,sha
from .resources import check_resources


def _attempt_window(window,attempt):
    if attempt.get('purpose')!='recover_missing_targets':
        return window
    position=window['attempts'].index(attempt)
    previous=window['attempts'][position-1] if position else {}
    error=previous.get('error','')
    prefix='存在未覆盖的target_spans：'
    if not error.startswith(prefix):
        raise ValueError('cannot reconstruct recorded recovery targets')
    missing=error[len(prefix):].split(',')
    ordered=window['context_ids']+window['target_ids'];context=[]
    if not missing or any(s not in window['target_ids'] for s in missing):
        raise ValueError('invalid recorded recovery targets')
    for span_id in missing:
        i=ordered.index(span_id);context.extend(ordered[max(0,i-2):i])
    return {'target_ids':missing,'context_ids':[s for s in dict.fromkeys(context) if s not in missing]}


def replay_archive(directory,output):
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError('use a new independent replay output directory')
    manifest=read(directory/'manifest.json')
    validator_path=Path(validator.__file__).resolve()
    frozen=[digest for path,digest in manifest.get('files',{}).items()
            if path.replace('\\','/').endswith('/agent_pipeline_v2/extractor.py')]
    if len(frozen)!=1 or frozen[0]!=sha(validator_path):
        raise ValueError('validator does not match frozen source archive')
    selected=[a for a in manifest['config']['arms'] if a in ('original_plain','original_lookup5')]
    if not selected:
        raise ValueError('replay requires original protocol baseline arms')
    paths=[directory/'manifest.json',Path(__file__),Path(__file__).with_name('repair.py'),validator_path]
    for arm in selected:
        base=directory/arm
        paths += [base/name for name in ('manifest.json','complete.json','extractor_runs.jsonl','inference_calls.jsonl','wire_calls.jsonl')]
    identity=build_identity({'policy':POLICY,'source_archive':str(directory)},paths)
    result={}
    for arm in selected:
        base=directory/arm;complete=read(base/'complete.json')
        if not complete.get('complete'):
            raise ValueError('replay requires complete archived arms')
        if set(complete['files'])!={'extractor_runs.jsonl','inference_calls.jsonl','wire_calls.jsonl'}:
            raise ValueError('source arm must bind all three call logs')
        for name,digest in complete['files'].items():
            if sha(base/name)!=digest:raise ValueError('source arm file changed')
        rows=jsonl(base/'extractor_runs.jsonl');calls=jsonl(base/'inference_calls.jsonl');wires=jsonl(base/'wire_calls.jsonl')
        if [r['case_id'] for r in rows]!=manifest['config']['case_ids'] or any(r.get('status')!='ok' for r in rows):
            raise ValueError('source arm has incomplete or failed stories')
        validate_arm_identity(directory,arm,rows);validate_logs(rows,calls,wires)
        refs={a['timing']['call_id']:(r,w,a) for r in rows for w in r['result']['benchmark_stages']['extraction']['calls'] for a in w['attempts']}
        examples=[]
        for wire in wires:
            _,actions=repair_raw(wire.get('raw_output'))
            if not actions:continue
            ref=refs.get(wire.get('call_id'));schema_error='no exact attempt reference';post_actions=[];schema_known=False
            if ref:
                row,window,attempt=ref
                repaired,post_actions=repair_raw(attempt['raw_output'])
                try:
                    import json
                    ex=row['result']['benchmark_stages']['extraction']
                    actual_window=_attempt_window(window,attempt)
                    schema_known=True
                    value,_=_normalize_window_value(json.loads(repaired),ex['spans'],actual_window)
                    _decode_window(value,ex['spans'],actual_window,1)
                    schema_error=None
                except (ValueError,TypeError,KeyError) as exc:
                    schema_error=str(exc)
            examples.append({'call_id':wire.get('call_id'),'case_id':ref[0]['case_id'] if ref else None,
                             'actions':actions,'recorded_post_repair_actions':post_actions,
                             'recorded_validation_error':ref[2].get('error') if ref else None,
                             'schema_error_after':schema_error,'schema_known':schema_known,'quality':None})
        result[arm]={'responses':len(wires),'changed_responses':len(examples),
                     'schema_valid_after':sum(e['schema_error_after'] is None for e in examples),
                     'schema_unknown':sum(not e['schema_known'] for e in examples),'examples':examples}
    check_resources(output,initial=True)
    if build_identity(identity['config'],paths)!=identity:
        raise ValueError('archive changed during replay')
    output.mkdir(parents=True,exist_ok=False)
    value={'arms':result,'notice':'Offline schema replay only; no speed/quality score or saved-call claim.'}
    atomic_write_json(output/'manifest.json',identity);atomic_write_json(output/'replay.json',value)
    lines=['# Benchmark 5C1 离线字段修复回放','','只验证已记录回复的字段修复与窗口校验，不加载模型、不测速度或质量。','',
           '|基线|回复|可应用白名单的回复|修复后通过窗口校验|窗口无法重建|','|---|---:|---:|---:|---:|']
    for arm,stats in result.items():
        lines.append(f"|{arm}|{stats['responses']}|{stats['changed_responses']}|{stats['schema_valid_after']}|{stats['schema_unknown']}|")
    lines+=['','原始模型回复用于定位白名单命中；窗口校验使用归档的旧repair-wrapper输出再修字段，以保留其处理痕迹。',
            '这不是重新执行生成和整个修复链。接受旧首轮可能保留更多过选/幻觉事件，不能由字段校验通过推断语义正确。',
            '真实对照仍需原/修复 × 普通/lookup同轮运行，全部事件重新审核；原5B归档保持不变。']
    (output/'replay.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPLAY='+str(output/'replay.md'),flush=True)
    return value
