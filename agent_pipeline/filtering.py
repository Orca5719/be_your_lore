"""Select constrained or consequential events without rewriting their semantics."""
import copy
import json
import time
from pathlib import Path
from .llm import call_json

EVENT_KEYS=('id','actors','event','mental_state','explicit','modality','conditions')
CONTEXT_EVENT_RADIUS=2
REASON_CODES={
    'routine':('ignore','普通日常或背景，无特殊影响'),
    'process_detail':('ignore','呈现或操作过程细节，无独立核对价值'),
    'mechanism':('keep','特殊机制或异常状态，需核对设定约束'),
    'knowledge_relation':('keep','涉及身份、人物知识或关系变化'),
    'state_time_space':('keep','涉及重要状态、时间或空间约束'),
    'consequence_support':('keep','重要后果或相关事件所需核对依据'),
    'unclear':('review','主体、依据或重要性待复核'),
}


def decode_decision_row(raw):
    if not isinstance(raw,dict) or set(raw)!={'event_id','decision','reason_code'}:
        raise ValueError('决定字段不合法；不得改写事件')
    code=raw['reason_code'];decision=raw['decision'];normalizations=[]
    if isinstance(code,str) and code in REASON_CODES and decision==code:
        decision=REASON_CODES[code][0];normalizations.append('decision_from_reason_code')
    if not isinstance(code,str) or code not in REASON_CODES or decision!=REASON_CODES[code][0]:
        raise ValueError('决定字段不合法；不得改写事件')
    row=dict(event_id=raw['event_id'],decision=decision,reason=REASON_CODES[code][1])
    if normalizations:row['wire_normalizations']=normalizations
    return row,code


def validate_input(report):
    if not isinstance(report,dict) or report.get('stage')!='understanding' or report.get('status') not in ('ok','partial','error'):
        raise ValueError('筛选输入必须是语义理解模块的完整JSON报告')
    events=report.get('events')
    if not isinstance(events,list) or len(events)>160:raise ValueError('events必须是数组，最多160项')
    ids=set()
    for event in events:
        if not isinstance(event,dict) or not all(key in event for key in EVENT_KEYS):raise ValueError('语义事件字段不完整')
        eid=event['id']
        if not isinstance(eid,str) or not eid.strip() or eid in ids:raise ValueError('事件ID必须是非空且唯一的文字')
        ids.add(eid)
        if not isinstance(event['event'],str) or not event['event'].strip():raise ValueError('event必须是非空文字')
        if not isinstance(event['actors'],list) or not event['actors'] or any(not isinstance(a,str) or not a.strip() for a in event['actors']):raise ValueError('actors必须是非空文字数组')
        if event['modality'] not in ('observed','speech','belief','plan','dream','inferred','conditional') or not isinstance(event['explicit'],bool):raise ValueError('叙述性质或explicit无效')
        if not isinstance(event['conditions'],list) or any(not isinstance(c,str) for c in event['conditions']):raise ValueError('conditions必须是文字数组')
        if event['mental_state'] is not None and not isinstance(event['mental_state'],str):raise ValueError('mental_state必须是文字或null')
        if not isinstance(event.get('review_reasons',[]),list) or any(not isinstance(reason,str) for reason in event.get('review_reasons',[])):
            raise ValueError('review_reasons必须是文字数组')
        for field in ('source_ids','context_ids'):
            if field in event and (not isinstance(event[field],list) or any(not isinstance(ref,str) or not ref.strip() for ref in event[field])):
                raise ValueError(field+'必须是编号文字数组')


def filter_events(understanding_report,device='auto',llm=None,progress=None):
    validate_input(understanding_report)
    if device not in ('auto','cpu','cuda'):raise ValueError('device必须是auto、cpu或cuda')
    started=time.perf_counter()
    upstream=copy.deepcopy(understanding_report)
    events=upstream['events']
    wire=[{key:event[key] for key in EVENT_KEYS+('source_ids','context_ids','sources','contexts','review_reasons') if key in event} for event in events]
    loaded_here=False
    calls=[];rejected=[];extraneous_decisions=[];decisions=[]
    if events:
        if llm is None:
            from qwen_judge import QwenJudge
            llm=QwenJudge(device);loaded_here=True
        prompt=(Path(__file__).parent/'prompts/filtering_v7.txt').read_text(encoding='utf-8')
        for offset in range(0,len(events),8):
            targets=events[offset:offset+8]
            target_ids={e['id'] for e in targets}
            end=offset+len(targets)
            context_rows=wire[max(0,offset-CONTEXT_EVENT_RADIUS):offset]+wire[end:min(len(wire),end+CONTEXT_EVENT_RADIUS)]
            context_events=[{key:event[key] for key in EVENT_KEYS if key in event} for event in context_rows]
            payload=dict(target_events=[e for e in wire if e['id'] in target_ids],context_events=context_events)
            if progress:progress(dict(window_id=len(calls)+1,purpose='filtering',target_ids=[e['id'] for e in targets]))
            def validate_top(value):
                if set(value)!={'decisions'} or not isinstance(value['decisions'],list) or len(value['decisions'])>40:
                    raise ValueError('只输出decisions数组，最多40项')
            value,call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],max_output=1024,validator=validate_top)
            call.update(window_id=len(calls)+1,target_ids=[e['id'] for e in targets],context_event_ids=[e['id'] for e in context_events]);calls.append(call)
            rows={};invalid=set();codes={};window_rejections=[]
            known_ids={event['id'] for event in events}
            if value:
                for index,row in enumerate(value['decisions'],1):
                    raw_row=row
                    decoded=None;code=None
                    try:decoded,code=decode_decision_row(raw_row)
                    except (ValueError,TypeError):pass
                    eid=raw_row.get('event_id') if isinstance(raw_row,dict) else None
                    error=None
                    if isinstance(eid,str) and eid in known_ids and eid not in target_ids:
                        extraneous_decisions.append(dict(window_id=call['window_id'],candidate_index=index,raw_decision=raw_row,error='已知但非当前目标事件ID，已忽略'))
                        continue
                    if not isinstance(eid,str) or eid not in target_ids:error='未知事件ID'
                    elif decoded is None:
                        error='决定字段不合法；不得改写事件'
                    elif eid in rows:error='事件决定重复或冲突'
                    if error:
                        record=dict(window_id=call['window_id'],candidate_index=index,raw_decision=raw_row,error=error,recovered_event_ids=[])
                        rejected.append(record);window_rejections.append(record)
                        if isinstance(eid,str) and eid in target_ids:invalid.add(eid)
                    else:
                        rows[eid]=decoded;codes[eid]=code
            unresolved={event['id'] for event in targets if event['id'] not in rows or event['id'] in invalid}
            if unresolved and call['status']=='ok':
                repair_payload=dict(target_events=[e for e in wire if e['id'] in unresolved],context_events=context_events,
                                    repair_instruction='只修复这些target_events的筛选决定；每个ID恰好一次，不输出其他事件。')
                def validate_repair(answer):
                    validate_top(answer)
                    seen=set()
                    for row in answer['decisions']:
                        decoded,code=decode_decision_row(row);eid=decoded['event_id']
                        if eid not in unresolved or eid in seen:raise ValueError('修复事件ID无效或重复')
                        seen.add(eid)
                    if seen!=unresolved:raise ValueError('修复结果未覆盖全部目标事件')
                if progress:progress(dict(window_id=len(calls)+1,purpose='filtering_repair',target_ids=sorted(unresolved)))
                repaired,repair_call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(repair_payload,ensure_ascii=False,separators=(',',':'))}],max_output=512,validator=validate_repair)
                repair_call.update(window_id=len(calls)+1,purpose='decision_repair',target_ids=sorted(unresolved),context_event_ids=[e['id'] for e in context_events]);calls.append(repair_call)
                if repaired:
                    for raw_row in repaired['decisions']:
                        normalized,code=decode_decision_row(raw_row);eid=normalized['event_id']
                        rows[eid]=normalized;codes[eid]=code
                    for record in window_rejections:
                        eid=record['raw_decision'].get('event_id') if isinstance(record['raw_decision'],dict) else None
                        if eid in rows:record['recovered_event_ids']=[eid]
                    invalid-=set(rows)
            for event in targets:
                eid=event['id']
                row=copy.deepcopy(rows[eid]) if eid in rows and eid not in invalid else dict(event_id=eid,decision='review',reason='筛选调用失败' if call['status']=='error' else '缺少有效且唯一的筛选决定',origin='program_fallback')
                if eid in codes and eid not in invalid:row['reason_code']=codes[eid]
                reasons=set(event.get('review_reasons',[]))
                if row['decision']=='ignore' and event['modality']=='inferred':
                    row.update(model_decision='ignore',model_reason=row['reason'],model_reason_code=row.get('reason_code'),reason_code='unclear',decision='review',reason='上游主体/推断依据待复核，暂不自动忽略',origin='program_guard')
                elif row['decision']=='keep' and 'unresolved_actor' in reasons:
                    row.update(model_decision='keep',model_reason=row['reason'],model_reason_code=row.get('reason_code'),reason_code='unclear',decision='review',reason='事件值得核对，但主体仍是未解析指代，不能送入后续判断',origin='program_guard')
                decisions.append(row)
    lookup={e['id']:e for e in events}
    # Preserve support required by a retained event; this does not prove causality.
    for _ in range(len(events)+1):
        kept=[lookup[d['event_id']] for d in decisions if d['decision']=='keep']
        changed=False
        for row in decisions:
            if row['decision']!='ignore':continue
            source_ids=set(lookup[row['event_id']].get('source_ids',[]))
            required_by=[e['id'] for e in kept if source_ids.intersection(e.get('context_ids',[]))]
            if required_by:
                row.update(model_decision='ignore',model_reason=row['reason'],model_reason_code=row.get('reason_code'),decision='keep',reason_code='consequence_support',support_only=True,origin='program_dependency_guard',required_by_event_ids=required_by,reason='保全保留事件的引用依据；不证明因果')
                changed=True
        if not changed:break
    pending=[lookup[d['event_id']] for d in decisions if d['decision']=='review']
    unresolved_rejected=[row for row in rejected if not row.get('recovered_event_ids')]
    complete=not unresolved_rejected and not pending and all(c['status']=='ok' for c in calls)
    failed=bool(calls) and all(c['status']=='error' for c in calls)
    status='error' if failed or upstream['status']=='error' else 'partial' if not complete or upstream['status']!='ok' else 'ok'
    failure_reasons=dict(upstream_status=upstream['status'],pending_event_ids=[e['id'] for e in pending],rejected_count=len(unresolved_rejected),rejected_total=len(rejected),failed_window_ids=[c['window_id'] for c in calls if c['status']=='error'])
    return dict(schema_version='agent-pipeline-filtering-v1',prompt_version='filtering-v7',stage='filtering',status=status,failure_reasons=failure_reasons,
                events=events,decisions=decisions,selected_events=[lookup[d['event_id']] for d in decisions if d['decision']=='keep'],
                ignored_events=[lookup[d['event_id']] for d in decisions if d['decision']=='ignore'],pending_events=pending,
                understanding=upstream,upstream_status=upstream['status'],calls=calls,rejected=rejected,extraneous_decisions=extraneous_decisions,
                filtering_complete=complete,processing_complete=status=='ok',semantic_verification='not_verified',
                request_seconds=time.perf_counter()-started,model_loaded_this_request=loaded_here,
                device=getattr(llm,'device',device),model_load_seconds=getattr(llm,'load_seconds',None),
                notice='仅筛选建议；事件原样保留，待复核不等于忽略；未检索、未判断矛盾、未保存设定。')
