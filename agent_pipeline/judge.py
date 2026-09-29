"""Per-event evidence-bound judgement; no story aggregation or canon writes."""
import copy
import json
import time
from pathlib import Path
from .llm import call_json
from .retrieval import validate_filtering

LABELS={'consistent':'明确吻合','contradiction':'明确矛盾','uncertain':'不确定'}
NONACTUAL={'speech','belief','plan','dream','inferred'}
UNCERTAINTY_REASONS={
    'insufficient':'本次候选证据不足以对这项事实给出明确判断，需核实适用设定。',
    'missing_rule':'本次候选设定未明确覆盖这项事实；未找到支持不等于设定禁止，需核实适用规则。',
    'ambiguous_reference':'这项事实与候选设定的对象或指代尚未明确对应，不能自行补入左右、身份等属性。',
    'unknown_conditions':'候选规则是否适用于当前时间、场景或条件尚不明确，需补充依据。',
    'partial_support':'候选证据仅支持这项命题的一部分，尚不足以确认完整命题。',
    'conflicting_evidence':'模型标记候选证据存在冲突，需作者核实原文及适用范围，暂不选定一方。',
}
NOTICE='逐事件一致性建议，未汇总整段、未保存设定。有效引用不保证判断正确；未提取/待复核内容不代表无矛盾。'


def validate_retrieval(report):
    if not isinstance(report,dict) or report.get('stage')!='retrieval' or report.get('status') not in ('ok','partial','error'):raise ValueError('judge输入须为检索模块完整JSON报告')
    upstream=report.get('filtering');rows=validate_filtering(upstream)
    if upstream['status']=='error' and report['status']!='error' or upstream['status']=='partial' and report['status']=='ok':raise ValueError('检索状态不能掩盖筛选未完成状态')
    for field,subset in [('pending_event_ids','pending_events'),('ignored_event_ids','ignored_events')]:
        if report.get(field)!=[e['id'] for e in upstream[subset]]:raise ValueError(field+'与上游分区不匹配或缺失')
    if report.get('events')!=upstream['events']:raise ValueError('检索事件与筛选原事件不匹配')
    items=report.get('items')
    if not isinstance(items,list):raise ValueError('items须为数组')
    selected={e['id']:e for e in upstream['selected_events']};seen=set()
    for item in items:
        if not isinstance(item,dict) or not isinstance(item.get('event_id'),str) or item['event_id'] not in selected or item['event_id'] in seen:raise ValueError('检索事件编号无效或重复')
        eid=item['event_id'];seen.add(eid)
        if item.get('event')!=selected[eid] or item.get('status') not in ('ok','error','support_only'):raise ValueError('检索事件或状态不匹配')
        if (item['status']=='support_only')!=bool(rows[eid].get('support_only')):raise ValueError('支持事件路由不匹配')
        evidence=item.get('evidence');ids=set()
        if not isinstance(evidence,list):raise ValueError('evidence须为数组')
        for chunk in evidence:
            if not isinstance(chunk,dict) or not isinstance(chunk.get('id'),str) or not chunk['id'] or chunk['id'] in ids or not isinstance(chunk.get('text'),str) or not chunk['text'].strip():raise ValueError('证据ID/原文无效或重复')
            ids.add(chunk['id'])
    if report['status']!='error' and seen!=set(selected):raise ValueError('缺少保留事件的检索记录')
    return rows


def validate_answer(value,evidence):
    expected={'verdict','citations','reason','assessment'}
    if set(value)-{'uncertainty_code'}!=expected:raise ValueError('判断字段无效，须有assessment/verdict/citations/reason，仅额外允许uncertainty_code')
    if value.get('uncertainty_code') is not None and (not isinstance(value['uncertainty_code'],str) or value['uncertainty_code'] not in UNCERTAINTY_REASONS):raise ValueError('uncertainty_code须为'+','.join(UNCERTAINTY_REASONS)+'或null')
    if not isinstance(value.get('verdict'),str) or value['verdict'] not in LABELS:raise ValueError('verdict须为consistent/contradiction/uncertain')
    assessment=value['assessment']
    if not isinstance(assessment,dict) or set(assessment)!={'same_subject','evidence_applicable','relation','assumptions'}:raise ValueError('assessment字段无效')
    if any(assessment[k] is not None and type(assessment[k]) is not bool for k in ('same_subject','evidence_applicable')):raise ValueError('对象/场景对应须为布尔或null')
    if assessment['relation'] not in ('direct_support','direct_conflict','insufficient'):raise ValueError('证据关系无效')
    if not isinstance(assessment['assumptions'],list) or any(not isinstance(x,str) or not x.strip() for x in assessment['assumptions']):raise ValueError('额外假设须为文字数组')
    if not isinstance(value['reason'],str) or not value['reason'].strip():raise ValueError('判断原因不能为空')
    citations=value['citations']
    if not isinstance(citations,list) or len(citations)>len(evidence):raise ValueError('证据引用须为有效数组')
    if value['verdict']!='uncertain' and not citations:raise ValueError('明确结论须引用设定')
    aliases={'L'+str(i):chunk for i,chunk in enumerate(evidence,1)};seen=set()
    for citation in citations:
        if not isinstance(citation,dict) or set(citation)!={'evidence_id','quote'}:raise ValueError('引用字段无效')
        label=citation['evidence_id'];quote=citation['quote']
        if not isinstance(label,str) or label not in aliases or label in seen:raise ValueError('引用编号不存在或重复；只能用本次'+','.join(aliases)+'，不能引用事件S/E编号')
        seen.add(label)
        if not isinstance(quote,str) or not quote.strip() or quote not in aliases[label]['text']:raise ValueError('引用原文不属于对应设定')


def judge_events(retrieval_report,device='auto',llm=None,progress=None):
    rows=validate_retrieval(retrieval_report)
    if device not in ('auto','cpu','cuda'):raise ValueError('device必须为auto/cpu/cuda')
    started=time.perf_counter();upstream=copy.deepcopy(retrieval_report);items=[];loaded=False;load_error=None
    prompt=(Path(__file__).parent/'prompts/judge_v6.txt').read_text(encoding='utf-8')
    if upstream['status']!='error':
        for source in upstream['items']:
            eid=source['event_id'];event=source['event']
            if source['status']=='support_only':
                items.append(dict(event_id=eid,status='support_only',event=event,required_by_event_ids=rows[eid]['required_by_event_ids'],citations=[]));continue
            item=dict(event_id=eid,event=event,evidence=source['evidence'],citations=[])
            if source['status']=='error':
                item.update(status='error',verdict='uncertain',label=LABELS['uncertain'],origin='retrieval_failure',error=source.get('error','检索失败'),reason='检索未成功，不能有效判断');items.append(item);continue
            if event['modality'] in NONACTUAL:
                item.update(status='ok',verdict='uncertain',label=LABELS['uncertain'],origin='program_nonactual_scope',reason='非客观叙述内容不按现实事件核对；本版尚未核实对应性质的设定约束');items.append(item);continue
            if not source['evidence']:
                item.update(status='ok',verdict='uncertain',label=LABELS['uncertain'],origin='program_no_evidence',reason='没有候选设定，无法明确核对');items.append(item);continue
            if llm is None and load_error is None:
                try:
                    from qwen_judge import QwenJudge
                    llm=QwenJudge(device);loaded=True
                except (ValueError,OSError,RuntimeError) as exc:load_error=str(exc)
            if load_error is not None:
                item.update(status='error',verdict='uncertain',label=LABELS['uncertain'],origin='model_load_failure',reason='本地模型加载失败，未完成判断',error=load_error);items.append(item);continue
            if progress:progress(dict(window_id=len(items)+1,purpose='judge',target_ids=[eid]))
            supports=[support['event'] for support in upstream['items'] if support['status']=='support_only' and eid in rows[support['event_id']]['required_by_event_ids']]
            lore=[dict(evidence_id='L'+str(i),text=chunk['text'],heading_path=chunk.get('heading_path',[])) for i,chunk in enumerate(source['evidence'],1)]
            payload=dict(event={k:event[k] for k in ('actors','event','mental_state','explicit','modality','conditions','sources','contexts','review_reasons') if k in event},support_events=supports,lore=lore)
            payload['story_context']=upstream['filtering'].get('understanding',{}).get('text','')
            value,call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],max_output=768,validator=lambda v:validate_answer(v,source['evidence']))
            item['call']=call
            if value is None:item.update(status='error',verdict='uncertain',label=LABELS['uncertain'],origin='judge_failure',reason='模型未返回有效判断',error=call['attempts'][-1]['error'])
            else:
                citations=[dict(citation,chunk_id=source['evidence'][int(citation['evidence_id'][1:])-1]['id']) for citation in value['citations']]
                item.update(status='ok',verdict=value['verdict'],label=LABELS[value['verdict']],reason=value['reason'],citations=citations,origin='model')
                item['assessment']=value['assessment']
                if value['verdict']=='uncertain':
                    code=value.get('uncertainty_code') or 'insufficient'
                    item.update(model_reason=value['reason'],uncertainty_code=code,reason=UNCERTAINTY_REASONS[code],reason_origin='program_uncertainty_code')
                assessment=value['assessment'];relation='direct_support' if value['verdict']=='consistent' else 'direct_conflict'
                if value['verdict']!='uncertain' and (assessment['same_subject'] is not True or assessment['evidence_applicable'] is not True or assessment['assumptions'] or assessment['relation']!=relation):
                    item.update(model_verdict=value['verdict'],verdict='uncertain',label=LABELS['uncertain'],origin='program_evidence_scope_guard',model_reason=value['reason'],reason='证据对象/适用场景未对应，或结论依赖额外假设，不能下明确结论')
            items.append(item)
    active=[x for x in items if x['status']!='support_only'];failed=[x['event_id'] for x in active if x['status']=='error']
    status='error' if upstream['status']=='error' or (active and len(failed)==len(active)) else 'partial' if failed or upstream['status']!='ok' else 'ok'
    stages={}
    for report in (upstream['filtering'].get('understanding',{}),upstream['filtering'],upstream):
        if report.get('status') in ('partial','error'):stages[report.get('stage','understanding')]=report.get('failure_reasons',dict(status=report['status']))
    return dict(schema_version='agent-pipeline-judge-v1',prompt_version='judge-v6',stage='judge',status=status,events=upstream['events'],items=items,retrieval=upstream,
                pending_event_ids=upstream.get('pending_event_ids',[]),ignored_event_ids=upstream.get('ignored_event_ids',[]),upstream_status=upstream['status'],failure_reasons=dict(upstream_status=upstream['status'],failed_event_ids=failed,upstream_stages=stages),
                judge_complete=not failed and upstream['status']!='error',processing_complete=status=='ok',semantic_verification='not_verified',model_loaded_this_request=loaded,device=getattr(llm,'device',device),model_load_seconds=getattr(llm,'load_seconds',None),request_seconds=time.perf_counter()-started,notice=NOTICE)
