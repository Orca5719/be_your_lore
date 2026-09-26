"""Stage one: all semantic events, with auditable original references."""
import re
import json
import time
from pathlib import Path
from .llm import call_json

FIELDS={'actors','event','mental_state','explicit','modality','conditions','source_ids','context_ids'}
MODALITIES={'observed','speech','belief','plan','dream','inferred'}
MODALITIES.add('conditional')
OPTIONAL_DEFAULTS={'mental_state':None,'conditions':[],'context_ids':[]}
PRONOUNS={'他','她','它','他们','她们','它们','那人','那个人','某人'}
CONTEXT_SPAN_LIMIT=2


def split_spans(text):
    spans={}
    for m in re.finditer(r'[^\r\n，,。！？!?；;]+[，,。！？!?；;]?',text):
        start=m.start()+len(m.group())-len(m.group().lstrip())
        end=m.end()-(len(m.group())-len(m.group().rstrip()))
        for segment_start in range(start,end,180):
            segment_end=min(segment_start+180,end)
            spans['S'+str(len(spans)+1)]=dict(text=text[segment_start:segment_end],start=segment_start,end=segment_end)
    return spans


def prior_context_ids(spans,first_target,limit=CONTEXT_SPAN_LIMIT):
    ids=list(spans);first=ids.index(first_target)
    return ids[max(0,first-limit):first]


def windows(spans):
    ids=list(spans);batches=[];target=[];chars=0
    for key in ids:
        size=len(spans[key]['text'])
        if target and (chars+size>180 or len(target)>=8):
            batches.append(dict(target_ids=target,context_ids=prior_context_ids(spans,target[0])))
            target=[];chars=0
        target.append(key);chars+=size
    if target:
        batches.append(dict(target_ids=target,context_ids=prior_context_ids(spans,target[0])))
    return batches


def validate_event(row,spans,batch):
    if not isinstance(row,dict) or set(row)!=FIELDS:raise ValueError('事件字段不符合协议')
    if not isinstance(row['event'],str) or not row['event'].strip():raise ValueError('event不能为空')
    for field in ['actors','conditions','source_ids','context_ids']:
        value=row[field]
        if not isinstance(value,list) or any(not isinstance(x,str) or not x.strip() for x in value):raise ValueError(field+'须为文字数组')
    if not row['actors'] or not row['source_ids']:raise ValueError('主体和原文依据不能为空')
    if row['mental_state'] is not None and (not isinstance(row['mental_state'],str) or not row['mental_state'].strip()):raise ValueError('mental_state须为文字或null')
    if type(row['explicit']) is not bool or row['modality'] not in MODALITIES:raise ValueError('叙述性质无效')
    if (not row['explicit'])!=(row['modality']=='inferred'):raise ValueError('推断必须标为explicit=false/modality=inferred')
    allowed=set(batch['target_ids']+batch['context_ids'])
    if any(x not in batch['target_ids'] for x in row['source_ids']) or any(x not in allowed for x in row['context_ids']):raise ValueError('引用不属于当前原文窗')
    if any(len(row[k])!=len(set(row[k])) for k in ['source_ids','context_ids']):raise ValueError('引用编号重复')
    referenced=[spans[k]['text'] for k in row['source_ids']+row['context_ids']]
    if any(not any(actor in text for text in referenced) for actor in row['actors']):
        raise ValueError('主体缺少原文依据：请选择含主体名称的source_ids/context_ids；指代不明则保留原文指代')
    return dict(row,sources=[dict(id=k,**spans[k]) for k in row['source_ids']],contexts=[dict(id=k,**spans[k]) for k in row['context_ids']])


def repair_actor_context(llm,rows,spans,batch):
    """Ask for references only. Never accept rewritten semantic fields."""
    available=set(batch['target_ids']+batch['context_ids'])
    candidates=[dict(row,actor_mentions={actor:[k for k,span in spans.items() if k in available and actor in span['text']] for actor in row['event']['actors']}) for row in rows]
    payload=dict(source_spans={k:span['text'] for k,span in spans.items() if k in available},candidates=candidates)
    prompt='只补充主体原文依据，不改写事件。根据source_spans给每个candidate选择含其actors名称且支持该指代的必要context_ids。指代不明不要补，不能只因名字出现就当作指代成立。只返回JSON {"repairs":[{"candidate_index":数字,"context_ids":["S1"]}]}。context_ids必须是字符串数组，逐字选择source_spans键（如"S1"），不能输出数字1。必须逐个覆盖输入的每个candidate_index；没有可靠依据的候选context_ids返回[]。不得增加其他字段或未给出的编号。'
    prompt+='actor_mentions列出每个主体名称逐字出现的候选依据；请从这些片段选可靠先行词，而非重复引用当前source。该表只证明名称出现，不证明指代正确。'
    value,call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],max_output=512)
    fixes={}
    if value is None:return fixes,call
    allowed={x['candidate_index'] for x in rows}
    if set(value)!={'repairs'} or not isinstance(value['repairs'],list):
        call.update(status='error',error='引用补充协议无效');return fixes,call
    try:
        for item in value['repairs']:
            if not isinstance(item,dict) or set(item)!={'candidate_index','context_ids'}:raise ValueError('引用补充不能改写语义字段')
            index=item['candidate_index'];contexts=item['context_ids']
            if type(index) is not int or index not in allowed or index in fixes:raise ValueError('补充候选编号无效或重复')
            if not isinstance(contexts,list) or any(not isinstance(x,str) or x not in spans for x in contexts):raise ValueError('补充引用编号无效')
            fixes[index]=contexts
    except ValueError as exc:call.update(status='error',error=str(exc));return {},call
    return fixes,call


def add_unique_name_anchors(row,spans,batch):
    """Locate names already proposed by the model; do not resolve new actors."""
    if not isinstance(row,dict) or set(row)!=FIELDS:return row
    if not isinstance(row['actors'],list) or not isinstance(row['source_ids'],list) or not isinstance(row['context_ids'],list):return row
    allowed=set(batch['target_ids']+batch['context_ids'])
    refs=row['source_ids']+row['context_ids']
    if any(not isinstance(k,str) or k not in allowed for k in refs) or not row['source_ids']:return row
    first=min(spans[k]['start'] for k in row['source_ids'])
    added=[]
    for actor in row['actors']:
        if not isinstance(actor,str) or not actor:continue
        if any(actor in spans[k]['text'] for k in refs+added):continue
        mentions=[k for k,span in spans.items() if k in allowed and span['end']<=first and actor in span['text']]
        if mentions:added.append(mentions[-1])
    return dict(row,context_ids=list(dict.fromkeys(row['context_ids']+added))) if added else row


def referenced_spans(events):
    return {sid for event in events for sid in event['source_ids']+event['context_ids']}


def validate_envelope(value,batch=None):
    if not isinstance(value,dict) or set(value)-{'events','non_event_span_ids'} or 'events' not in value or not isinstance(value['events'],list) or len(value['events'])>40:
        raise ValueError('输出必须包含最多40条events，可附non_event_span_ids')
    non_event_ids=value.get('non_event_span_ids',[])
    if not isinstance(non_event_ids,list) or any(not isinstance(sid,str) for sid in non_event_ids) or len(non_event_ids)!=len(set(non_event_ids)):
        raise ValueError('non_event_span_ids必须是不重复的编号数组')
    if batch is not None:
        targets=set(batch['target_ids'])
        if any(sid not in targets for sid in non_event_ids):
            raise ValueError('non_event_span_ids只能引用当前target_spans')
        event_sources={sid for row in value['events'] if isinstance(row,dict) for sid in row.get('source_ids',[]) if isinstance(sid,str)}
        if event_sources.intersection(non_event_ids):
            raise ValueError('同一片段不能同时声明为事件依据和非事件片段')


def normalize_wire_row(raw):
    if not isinstance(raw,dict):return raw,[],[]
    row=dict(raw);normalizations=[];defaulted=[]
    for key in list(row):
        canonical=next((field for field in FIELDS if field.casefold()==key.casefold()),None)
        if canonical and canonical!=key and canonical not in row:
            row[canonical]=row.pop(key);normalizations.append('field_case_normalized:'+key)
    for field in ('actors','conditions','source_ids','context_ids'):
        if isinstance(row.get(field),str) and row[field].strip():
            row[field]=[row[field]];normalizations.append(field+'_scalar_wrapped')
    for key,default in OPTIONAL_DEFAULTS.items():
        if key not in row:
            row[key]=list(default) if isinstance(default,list) else default
            defaulted.append(key)
    return row,defaulted,normalizations


def understand(text,device='auto',llm=None,progress=None):
    started=time.perf_counter()
    if not isinstance(text,str) or not text.strip():raise ValueError('输入不能为空')
    if len(text)>800:raise ValueError('第一步目前支持不超过800字符；不会截断')
    if device not in ('auto','cpu','cuda'):raise ValueError('device只能是auto/cpu/cuda')
    spans=split_spans(text)
    if not spans:raise ValueError('没有可理解的文字片段')
    loaded_here=llm is None
    if loaded_here:
        from qwen_judge import QwenJudge
        llm=QwenJudge(device)
    prompt=(Path(__file__).parent/'prompts/understanding_v8.txt').read_text(encoding='utf-8')
    events=[];calls=[];rejected=[];extraneous_candidates=[];non_event_ids=[]
    def process_batch(batch,purpose):
        window_id=len(calls)+1
        if progress is not None:progress(dict(window_id=window_id,purpose=purpose,target_ids=batch['target_ids']))
        payload={field:{k:spans[k]['text'] for k in batch[ids]} for field,ids in [('target_spans','target_ids'),('context_spans','context_ids')]}
        relevant_rejected=[]
        if purpose=='coverage_recovery':
            target_ids=set(batch['target_ids'])
            for record in rejected:
                raw,_,_=normalize_wire_row(record['raw_event'])
                refs=[]
                if isinstance(raw,dict):
                    for field in ('source_ids','context_ids'):
                        value=raw.get(field,[])
                        if isinstance(value,list):refs.extend(x for x in value if isinstance(x,str))
                if target_ids.intersection(refs):relevant_rejected.append(record)
            payload['rejected_candidates']=[dict(candidate_id=r['candidate_id'],error=r['error'],event=r['raw_event']) for r in relevant_rejected]
            payload['recovery_instruction']='修正与target_spans有关的失败候选时，在新事件加入replaces:[对应candidate_id]，明确替代旧候选。保留全部原候选有效source_ids作为新事件source_ids，不能只用作context。修正主体/命题必须回到原文，不因旧候选出现就相信它。不相关、不能可靠修正的候选不替代。context事件不重复生成。'
        active_prompt=prompt
        if purpose=='coverage_recovery':
            active_prompt+='\n本次是纠错补提。每个修正事件额外允许且必须写replaces:[被替代的rejected_candidates.candidate_id]，新事件可纠正错误主体与表述，但不能引入原文没有的语义；旧候选只是错误输出，不是可信原文。新增遗漏事件无旧候选则replaces:[]。只为target_spans提事件；source_ids只能从target_spans取，前文context_spans编号只能放context_ids，不能重复提取前文事件或把前文人物操作拼进屏幕/环境事件。修正旧候选必须完整保留其中属于原文的source_ids。'
        messages=[{'role':'system','content':active_prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
        value,call=call_json(llm,messages,validator=lambda answer:validate_envelope(answer,batch))
        call.update(window_id=window_id,purpose=purpose,**batch);calls.append(call)
        if call['status']!='ok':return
        non_event_ids.extend(value.get('non_event_span_ids',[]))
        pending=[]
        for index,row in enumerate(value['events'],1):
            original=row
            replacements=row.get('replaces',[]) if isinstance(row,dict) else []
            if isinstance(row,dict) and 'replaces' in row:row={k:v for k,v in row.items() if k!='replaces'}
            row,defaulted,normalizations=normalize_wire_row(row)
            before_anchors=row
            row=add_unique_name_anchors(row,spans,batch)
            try:
                event=validate_event(row,spans,batch)
                if not isinstance(replacements,list) or any(not isinstance(x,str) for x in replacements) or len(replacements)!=len(set(replacements)):raise ValueError('replaces须为不重复候选编号数组')
                known={r['candidate_id']:r for r in relevant_rejected}
                for cid in replacements:
                    if purpose!='coverage_recovery' or cid not in known:raise ValueError('替代候选编号不属于当前补提')
                    old,_,_=normalize_wire_row(known[cid]['raw_event'])
                    refs=old.get('source_ids',[]) if isinstance(old,dict) else []
                    valid_refs=[k for k in refs if isinstance(k,str) and k in spans] if isinstance(refs,list) else []
                    if not valid_refs or any(k not in event['source_ids'] for k in valid_refs):raise ValueError('替代事件须完整保留旧候选有效原文依据')
                if replacements:event['replaces']=replacements
                if defaulted:event['defaulted_fields']=defaulted
                if normalizations:event['wire_normalizations']=normalizations
                if row!=before_anchors:
                    first=min(spans[k]['start'] for k in row['source_ids'])
                    multiple=any(sum(a in spans[k]['text'] for k in batch['target_ids']+batch['context_ids'] if spans[k]['end']<=first)>1 for a in row['actors'])
                    event.update(reference_repairs=['nearest_prior_name_anchor_added' if multiple else 'unique_prior_name_anchor_added'],original_context_ids=before_anchors['context_ids'],actor_grounding_note='主体由模型提出；程序只补名称出现位置，不证明指代理解正确')
                event['id']='E'+str(len(events)+1);events.append(event)
            except (ValueError,TypeError) as exc:
                source_ids=row.get('source_ids',[]) if isinstance(row,dict) else []
                if purpose=='coverage_recovery' and not replacements and source_ids and all(sid in batch['context_ids'] for sid in source_ids):
                    extraneous_candidates.append(dict(window_id=window_id,candidate_index=index,reason='coverage_recovery_context_only_output',raw_event=original))
                    continue
                record=dict(window_id=window_id,candidate_index=index,candidate_id=f'W{window_id}C{index}',error=str(exc),raw_event=original)
                available=[spans[k]['text'] for k in batch['target_ids']+batch['context_ids']]
                if str(exc).startswith('主体缺少原文依据') and all(any(a in s for s in available) for a in row['actors']):pending.append(record)
                else:rejected.append(record)
        if pending:
            fixes,repair_call=repair_actor_context(llm,[dict(candidate_index=x['candidate_index'],event=x['raw_event']) for x in pending],spans,batch)
            call['reference_repair']=repair_call
            for record in pending:
                original=record['raw_event'];index=record['candidate_index']
                if index not in fixes:rejected.append(record);continue
                normalized,defaulted,normalizations=normalize_wire_row(original)
                contexts=list(dict.fromkeys(normalized['context_ids']+fixes[index]))
                try:
                    repaired=validate_event(dict(normalized,context_ids=contexts),spans,batch)
                    repaired.update(id='E'+str(len(events)+1),reference_repairs=['missing_actor_context_added'],original_context_ids=normalized['context_ids'])
                    if defaulted:repaired['defaulted_fields']=defaulted
                    if normalizations:repaired['wire_normalizations']=normalizations
                    events.append(repaired)
                except (ValueError,TypeError) as exc:
                    rejected.append(dict(record,repair_error=str(exc)))
    for batch in windows(spans):process_batch(batch,'initial_understanding')
    covered=referenced_spans(events)|set(non_event_ids)
    blocked={sid for c in calls if c['status']=='error' and c['attempts'][-1].get('error_type') in ('RuntimeError','OSError') for sid in c['target_ids']}
    rejected_targets=set()
    for record in rejected:
        normalized,_,_=normalize_wire_row(record['raw_event'])
        if isinstance(normalized,dict):
            refs=normalized.get('source_ids') or normalized.get('context_ids',[])
            if isinstance(refs,list):rejected_targets.update(k for k in refs if isinstance(k,str) and k in spans)
    gaps=[k for k in spans if (k not in covered or k in rejected_targets) and k not in blocked]
    for batch in windows({k:spans[k] for k in gaps}):
        batch['context_ids']=prior_context_ids(spans,batch['target_ids'][0])
        process_batch(batch,'coverage_recovery')
    unique={};duplicate_candidates=0
    for event in events:
        # Only exact semantic fields and identical supporting text are duplicates.
        identity=json.dumps({k:event[k] for k in ['actors','event','mental_state','explicit','modality','conditions']},ensure_ascii=False,sort_keys=True)+json.dumps([(s['start'],s['end']) for s in event['sources']])
        if identity in unique:
            duplicate_candidates+=1
            previous=unique[identity]
            previous['replaces']=list(dict.fromkeys(previous.get('replaces',[])+event.get('replaces',[])))
            for field in ('source_ids','context_ids'):
                previous[field]=sorted(set(previous[field]+event[field]),key=lambda k:spans[k]['start'])
            previous['sources']=[dict(id=k,**spans[k]) for k in previous['source_ids']]
            previous['contexts']=[dict(id=k,**spans[k]) for k in previous['context_ids']]
        else:unique[identity]=event
    events=sorted(unique.values(),key=lambda e:min(spans[k]['start'] for k in e['source_ids']))
    for index,event in enumerate(events,1):
        event['id']='E'+str(index)
        issues=[]
        if any(a in PRONOUNS for a in event['actors']):issues.append('unresolved_actor')
        if event['modality']=='inferred':issues.append('inferred_content')
        if any(not any(a in span['text'] for span in event['sources']) for a in event['actors']):issues.append('actor_resolution_proposed_by_model')
        if event.get('defaulted_fields'):issues.append('optional_fields_defaulted')
        if event.get('wire_normalizations'):issues.append('wire_format_normalized')
        if event.get('replaces'):issues.append('recovered_candidate_reinterpreted')
        event['review_required']=bool(issues);event['review_reasons']=issues
    primary={sid for event in events for sid in event['source_ids']}
    non_event_ids=list(dict.fromkeys(non_event_ids))
    non_event_conflicts=[sid for sid in non_event_ids if sid in primary]
    covered=referenced_spans(events)|set(non_event_ids)
    missing=[sid for sid in spans if sid not in covered]
    semantic_keys=['actors','event','mental_state','explicit','modality','conditions','source_ids']
    unresolved_rejected=[]
    for record in rejected:
        raw,_,_=normalize_wire_row(record['raw_event'])
        keys=semantic_keys if isinstance(raw,dict) and isinstance(raw.get('modality'),str) and raw['modality'] in MODALITIES else [k for k in semantic_keys if k!='modality']
        matches=[e['id'] for e in events if isinstance(raw,dict) and all(raw.get(k)==e[k] for k in keys)]
        mapped=[e['id'] for e in events if record['candidate_id'] in e.get('replaces',[])]
        if mapped:record['recovery_method']='model_replacement_reference_validated';matches=list(dict.fromkeys(matches+mapped))
        record['recovered_event_ids']=matches
        if not matches:unresolved_rejected.append(record)
    failed_sources={sid for call in calls if call['status']=='error' for sid in call['target_ids'] if sid not in primary}
    for call in calls:
        if call['status']=='error':call['recovered_by_primary_sources']=all(sid in primary for sid in call['target_ids'])
    status='error' if not events and not non_event_ids and all(c['status']=='error' for c in calls) else 'partial' if unresolved_rejected or missing or failed_sources or non_event_conflicts else 'ok'
    return dict(schema_version='agent-pipeline-understanding-v1',prompt_version='understanding-v8',stage='understanding',status=status,text=text,events=events,non_event_spans=[dict(id=k,**spans[k]) for k in non_event_ids],rejected=rejected,extraneous_candidates=extraneous_candidates,calls=calls,duplicate_candidates=duplicate_candidates,unrecovered_failed_source_ids=[k for k in spans if k in failed_sources],failure_reasons=dict(unresolved_candidates=[dict(candidate_id=r['candidate_id'],error=r['error'],source_ids=r['raw_event'].get('source_ids',[]) if isinstance(r['raw_event'],dict) else []) for r in unresolved_rejected],missing_source_ids=missing,failed_source_ids=[k for k in spans if k in failed_sources],non_event_conflicts=non_event_conflicts),processing_complete=status=='ok',semantic_verification='not_verified',review_required=status!='ok' or bool(referenced_spans(events)-primary) or any(e['review_required'] for e in events),request_seconds=time.perf_counter()-started,model_loaded_this_request=loaded_here,
                coverage=dict(source_spans=spans,covered_source_ids=[k for k in spans if k in covered],primary_source_ids=[k for k in spans if k in primary],context_only_source_ids=[k for k in spans if k in referenced_spans(events)-primary],non_event_source_ids=[k for k in spans if k in non_event_ids],missing_source_ids=missing,note='包括事件/条件/主体上下文引用和显式非事件片段；被引用不保证全部语义提取正确'),
                device=llm.device,model_load_seconds=getattr(llm,'load_seconds',None),notice='仅语义理解：未筛选重要性、未检索、未判断、未保存设定')
