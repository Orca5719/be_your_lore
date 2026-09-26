"""Separate within-story check. Story citations never become canon evidence."""
import itertools
import json
from pathlib import Path
from .llm import call_json


def citation_texts(event):
    spans=event.get('sources',[])+event.get('contexts',[])
    texts=[s['text'] for s in spans]
    located={(s['start'],s['end'],s['text']) for s in spans if type(s.get('start')) is int and type(s.get('end')) is int and s['end']-s['start']==len(s['text'])}
    run='';end=None
    for start,stop,text in sorted(located):
        run=run+text if start==end else text
        texts.append(run);end=stop
    return texts


def validate_check_row(row,pid,pair):
    if not isinstance(row,dict) or set(row)!={'pair_id','verdict','same_subject','same_scope','assumptions','reason','citations'}:raise ValueError('故事检查字段无效')
    if row['pair_id']!=pid:raise ValueError('pair编号无效')
    if row['verdict'] not in ('contradiction','no_conflict','uncertain'):raise ValueError('故事判断类别无效')
    if any(row[k] is not None and type(row[k]) is not bool for k in ('same_subject','same_scope')):raise ValueError('对应关系须布尔或null')
    if not isinstance(row['reason'],str) or not row['reason'].strip() or not isinstance(row['assumptions'],list) or any(not isinstance(x,str) for x in row['assumptions']):raise ValueError('理由/假设无效')
    if not isinstance(row['citations'],list):raise ValueError('故事引用须数组')
    row=dict(row)
    if row['verdict']!='contradiction':
        if row['citations']:
            row['citations']=[];row['wire_normalizations']=['non_contradiction_citations_removed']
        return row
    event_map={e['id']:e for e in pair};cited=set()
    for citation in row['citations']:
        if not isinstance(citation,dict) or set(citation)!={'event_id','quote'} or not isinstance(citation['event_id'],str) or citation['event_id'] not in event_map:raise ValueError('故事引用编号无效')
        event=event_map[citation['event_id']]
        if not isinstance(citation['quote'],str) or not citation['quote'].strip() or not any(citation['quote'] in text for text in citation_texts(event)):raise ValueError('故事引用不是对应原文')
        cited.add(citation['event_id'])
    if row['verdict']=='contradiction' and cited!=set(event_map):raise ValueError('段内矛盾必须引用两方原文')
    return row


def check_story(understanding,llm=None,device='auto',progress=None,max_pairs=120):
    if type(max_pairs) is not int or max_pairs<0:raise ValueError('max_pairs须为非负整数')
    if understanding.get('stage')=='filtering':
        from .retrieval import validate_filtering
        rows=validate_filtering(understanding)
        all_events=understanding['events']
        events=[e for e in understanding['selected_events'] if e['modality']=='observed' and not rows[e['id']].get('support_only')]
        story_text=understanding.get('understanding',{}).get('text','')
        scope='filtered_world_relevant_observed_events'
    else:
        all_events=understanding['events'];events=[e for e in all_events if e['modality']=='observed']
        story_text=understanding.get('text','');scope='provided_observed_events'
    pairs=list(itertools.combinations(events,2));items=[];calls=[];rejected=[]
    targets=pairs[:max_pairs];load_error=None
    prompt=(Path(__file__).parent/'prompts/story_check_v1.txt').read_text(encoding='utf-8')
    for offset in range(0,len(targets),4):
        window=targets[offset:offset+4]
        rows={'P'+str(offset+i+1):pair for i,pair in enumerate(window)}
        def validate_top(value):
            if set(value)!={'checks'} or not isinstance(value['checks'],list) or len(value['checks'])>40:raise ValueError('只输出checks数组，最多40项')
        if llm is None and load_error is None:
            try:
                from qwen_judge import QwenJudge
                llm=QwenJudge(device)
            except (ValueError,OSError,RuntimeError) as exc:load_error=str(exc)
        if progress:progress(dict(window_id=len(calls)+1,purpose='story_check'))
        if load_error:
            value=None;call=dict(status='error',attempts=[dict(error=load_error)])
        else:
            payload=dict(story_text=story_text,pairs=[dict(pair_id=pid,events=list(pair)) for pid,pair in rows.items()])
            value,call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))}],max_output=1536,validator=validate_top)
        call.update(window_id=len(calls)+1,purpose='story_check',pair_ids=list(rows));calls.append(call)
        answers={};invalid=set();window_rejections=[]
        if value:
            for index,row in enumerate(value['checks'],1):
                pid=row.get('pair_id') if isinstance(row,dict) else None
                try:
                    if not isinstance(pid,str) or pid not in rows:raise ValueError('pair编号无效')
                    if pid in answers:raise ValueError('pair编号重复')
                    answers[pid]=validate_check_row(row,pid,rows[pid])
                except (ValueError,TypeError) as exc:
                    if isinstance(pid,str) and pid in rows:
                        invalid.add(pid);answers.pop(pid,None)
                    record=dict(window_id=call['window_id'],candidate_index=index,raw_check=row,error=str(exc),recovered_pair_ids=[])
                    rejected.append(record);window_rejections.append(record)
        unresolved=[pid for pid in rows if pid not in answers or pid in invalid]
        pair_errors={pid:call['attempts'][-1].get('error','缺少有效且唯一的故事检查结果') for pid in unresolved}
        if unresolved and call['status']=='ok':
            for pid in unresolved:
                pair=rows[pid]
                repair_payload=dict(story_text=story_text,pairs=[dict(pair_id=pid,events=list(pair))],repair_instruction='只修复这一对事件，返回且仅返回该pair_id的一条check。')
                def validate_repair(answer,pair_id=pid,pair_events=pair):
                    validate_top(answer)
                    if len(answer['checks'])!=1:raise ValueError('修复必须只包含当前pair的一条判断')
                    validate_check_row(answer['checks'][0],pair_id,pair_events)
                if progress:progress(dict(window_id=len(calls)+1,purpose='story_check_repair',pair_ids=[pid]))
                repaired,repair_call=call_json(llm,[{'role':'system','content':prompt},{'role':'user','content':json.dumps(repair_payload,ensure_ascii=False,separators=(',',':'))}],max_output=512,validator=validate_repair)
                repair_call.update(window_id=len(calls)+1,purpose='story_check_repair',pair_ids=[pid]);calls.append(repair_call)
                if repaired:
                    answers[pid]=validate_check_row(repaired['checks'][0],pid,pair);invalid.discard(pid);pair_errors.pop(pid,None)
                    for record in window_rejections:
                        raw_pid=record['raw_check'].get('pair_id') if isinstance(record['raw_check'],dict) else None
                        if raw_pid==pid:record['recovered_pair_ids']=[pid]
                else:pair_errors[pid]=repair_call['attempts'][-1].get('error','故事内检查修复失败')
        for pid,pair in rows.items():
            item=dict(pair_id=pid,event_ids=[e['id'] for e in pair],events=list(pair))
            if pid not in answers:item.update(status='error',verdict='uncertain',reason='故事内检查失败',error=pair_errors.get(pid,'缺少有效且唯一的故事检查结果'),citations=[])
            else:
                row=answers[pid];item.update(row,status='ok')
                if row['verdict']=='contradiction' and (row['same_subject'] is not True or row['same_scope'] is not True or row['assumptions']):
                    item.update(model_verdict='contradiction',model_reason=row['reason'],verdict='uncertain',reason='对象/时间/条件未对应或依赖额外假设，不能认定段内矛盾')
            items.append(item)
    missing=len(pairs)-len(targets)
    status='partial' if missing or any(i['status']=='error' for i in items) or understanding.get('status')!='ok' else 'ok'
    return dict(stage='story_check',schema_version='agent-pipeline-story-check-v1',status=status,scope=scope,
                candidate_event_ids=[e['id'] for e in events],total_observed_event_count=sum(e['modality']=='observed' for e in all_events),
                filtered_out_event_count=sum(e['modality']=='observed' for e in all_events)-len(events),
                pair_count=len(pairs),unprocessed_pair_count=missing,items=items,calls=calls,rejected=rejected,semantic_verification='not_verified')


def validate_story_result(result,events,expected_event_ids=None):
    if not isinstance(result,dict) or result.get('stage')!='story_check' or result.get('status') not in ('ok','partial'):raise ValueError('段内检查报告无效')
    all_observed={e['id']:e for e in events if e['modality']=='observed'}
    ids=result.get('candidate_event_ids')
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or any(not isinstance(eid,str) or eid not in all_observed for eid in ids):raise ValueError('段内候选事件编号无效')
    if expected_event_ids is not None and ids!=expected_event_ids:raise ValueError('段内候选未复用筛选结果')
    observed={eid:all_observed[eid] for eid in ids}
    count=len(observed)*(len(observed)-1)//2
    if result.get('total_observed_event_count')!=len(all_observed) or result.get('filtered_out_event_count')!=len(all_observed)-len(observed):raise ValueError('段内筛选计数无效')
    if result.get('pair_count')!=count or type(result.get('unprocessed_pair_count')) is not int or not 0<=result['unprocessed_pair_count']<=count:raise ValueError('段内检查数量无效')
    items=result.get('items');seen=set();pairs=set()
    if not isinstance(items,list) or len(items)+result['unprocessed_pair_count']!=count:raise ValueError('段内检查记录缺失')
    for item in items:
        if not isinstance(item,dict):raise ValueError('段内检查记录须为对象')
        ids=item.get('event_ids');pid=item.get('pair_id')
        if not isinstance(ids,list) or len(ids)!=2 or any(not isinstance(eid,str) or eid not in observed for eid in ids) or ids[0]==ids[1]:raise ValueError('段内事件编号无效')
        key=tuple(sorted(ids))
        if not isinstance(pid,str) or pid in seen or key in pairs or item.get('events')!=[observed[eid] for eid in ids]:raise ValueError('段内事件重复或内容不匹配')
        seen.add(pid);pairs.add(key)
        if item.get('status') not in ('ok','error') or item.get('verdict') not in ('contradiction','no_conflict','uncertain') or not isinstance(item.get('reason'),str):raise ValueError('段内判断无效')
        if item['status']=='error' and (item['verdict']!='uncertain' or not isinstance(item.get('error'),str)):raise ValueError('段内失败不能掩盖')
        citations=item.get('citations');cited=set()
        if not isinstance(citations,list):raise ValueError('段内引用无效')
        for c in citations:
            if not isinstance(c,dict) or set(c)!={'event_id','quote'} or not isinstance(c['event_id'],str) or c['event_id'] not in ids or not isinstance(c['quote'],str) or not c['quote'].strip():raise ValueError('段内引用编号无效')
            e=observed[c['event_id']]
            if not any(c['quote'] in text for text in citation_texts(e)):raise ValueError('段内引用不是原文')
            cited.add(c['event_id'])
        if item['verdict']=='contradiction' and (cited!=set(ids) or item.get('same_subject') is not True or item.get('same_scope') is not True or item.get('assumptions')!=[]):raise ValueError('段内矛盾依据不足')
    if result['status']=='ok' and (result['unprocessed_pair_count'] or any(i['status']=='error' for i in items)):raise ValueError('段内成功状态不能掩盖遗漏/失败')
