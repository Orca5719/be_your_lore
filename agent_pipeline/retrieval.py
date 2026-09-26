"""Embedding-only event retrieval; preserve upstream decisions and provenance."""
import copy
import json
import time
from pathlib import Path
from .filtering import validate_input

ROOT=Path(__file__).resolve().parent.parent
NOTICE='仅检索相关设定，未判断矛盾、未保存。相似度表示相对相关性，不是事实正确率或矛盾置信度；无关事件也可能返回片段。'


def event_query(event):
    parts=['主体：'+'、'.join(event['actors']),'叙述性质：'+event['modality'],'事件：'+event['event']]
    if event.get('mental_state'):parts.append('心理：'+event['mental_state'])
    if event['conditions']:parts.append('条件：'+'；'.join(event['conditions']))
    quotes=[]
    spans=[span for field in ('contexts','sources') for span in event.get(field,[])]
    spans.sort(key=lambda span:span['start'] if type(span.get('start')) is int else float('inf'))
    for span in spans:
        if span['text'] not in quotes:quotes.append(span['text'])
    if quotes:parts.append('局部原文：'+''.join(quotes))
    return '\n'.join(parts)


def validate_filtering(report):
    if not isinstance(report,dict) or report.get('stage')!='filtering':raise ValueError('检索输入必须是事件筛选模块完整JSON报告')
    validate_input(dict(report,stage='understanding'))
    events={e['id']:e for e in report['events']}
    decisions=report.get('decisions')
    if not isinstance(decisions,list) or len(decisions)!=len(events):raise ValueError('筛选决定数量与事件不匹配')
    rows={}
    for row in decisions:
        if not isinstance(row,dict) or not isinstance(row.get('event_id'),str) or row['event_id'] not in events or row['event_id'] in rows or row.get('decision') not in ('keep','ignore','review'):
            raise ValueError('筛选决定ID无效、重复或类别不合法')
        if 'support_only' in row and type(row['support_only']) is not bool:raise ValueError('support_only必须是布尔值')
        rows[row['event_id']]=row
    for name,choice in [('selected_events','keep'),('ignored_events','ignore'),('pending_events','review')]:
        subset=report.get(name)
        if not isinstance(subset,list) or any(not isinstance(e,dict) or not isinstance(e.get('id'),str) for e in subset):raise ValueError(name+'须为事件数组')
        ids=[e['id'] for e in subset]
        if len(set(ids))!=len(ids) or set(ids)!={eid for eid,row in rows.items() if row['decision']==choice} or any(e!=events.get(e['id']) for e in subset):
            raise ValueError(name+'与事件/决定不匹配')
    for row in rows.values():
        if row.get('support_only'):
            refs=row.get('required_by_event_ids')
            if row['decision']!='keep' or not isinstance(refs,list) or not refs or any(not isinstance(eid,str) or eid not in rows or rows[eid]['decision']!='keep' for eid in refs):
                raise ValueError('支持事件须引用有效保留事件')
    for event in events.values():
        for field in ('sources','contexts'):
            if field in event and (not isinstance(event[field],list) or any(not isinstance(span,dict) or not isinstance(span.get('text'),str) for span in event[field])):
                raise ValueError(field+'须包含有效原文文字')
    return rows


def retrieve_events(filtering_report,device='auto',index_directory=None,k=5,retriever=None,progress=None):
    rows=validate_filtering(filtering_report)
    if isinstance(k,bool) or not isinstance(k,int) or k<1:raise ValueError('k必须为正整数')
    if device not in ('auto','cpu','cuda'):raise ValueError('device必须为auto/cpu/cuda')
    started=time.perf_counter();upstream=copy.deepcopy(filtering_report)
    selected=upstream['selected_events'];items=[];cache={};loaded=False;index_info=None;load_error=None
    active=[e for e in selected if not rows[e['id']].get('support_only')]
    if upstream['status']!='error' and active and retriever is None:
        try:
            from index_store import load_index
            path=Path(index_directory) if index_directory is not None else ROOT/'data/index'
            # Fail cheaply for a missing/corrupt index before loading model weights.
            _,source_metadata=load_index(path)
            from .freshness import validate_sources
            validate_sources(source_metadata)
            from encoder import Encoder
            from retrieval import Retriever
            encoder=Encoder(device=device,offline=True,precision='float32')
            retriever=Retriever(path,encoder);loaded=True
        except (ValueError,OSError,RuntimeError) as exc:load_error=str(exc)
    if retriever is not None and hasattr(retriever,'metadata'):
        metadata=retriever.metadata
        index_info={key:copy.deepcopy(metadata[key]) for key in ('config','sources','lore_directory') if key in metadata}
    if upstream['status']!='error':
        for event in selected:
            eid=event['id'];row=rows[eid]
            if row.get('support_only'):
                items.append(dict(event_id=eid,event=event,status='support_only',required_by_event_ids=row['required_by_event_ids'],evidence=[],query=None,retrieval_seconds=0))
                continue
            query=event_query(event);t0=time.perf_counter()
            if progress:progress(dict(window_id=len(items)+1,purpose='retrieval',target_ids=[eid]))
            item=dict(event_id=eid,event=event,query=query,evidence=[],cache_hit=query in cache)
            try:
                if load_error:raise ValueError(load_error)
                if query not in cache:cache[query]=retriever.search(query,k=k)
                item.update(status='ok',evidence=copy.deepcopy(cache[query]))
            except (ValueError,OSError,RuntimeError) as exc:item.update(status='error',error=str(exc))
            item['retrieval_seconds']=time.perf_counter()-t0;items.append(item)
    failures=[x['event_id'] for x in items if x['status']=='error']
    all_failed=bool(active) and len(failures)==len(active)
    status='error' if upstream['status']=='error' or all_failed else 'partial' if failures or upstream['status']!='ok' else 'ok'
    return dict(schema_version='agent-pipeline-retrieval-v1',stage='retrieval',status=status,filtering=upstream,events=upstream['events'],items=items,top_k=k,
                pending_event_ids=[e['id'] for e in upstream['pending_events']],ignored_event_ids=[e['id'] for e in upstream['ignored_events']],
                upstream_status=upstream['status'],retrieval_complete=not failures and upstream['status']!='error',processing_complete=status=='ok',semantic_verification='not_verified',
                failure_reasons=dict(upstream_status=upstream['status'],failed_event_ids=failures,load_error=load_error),
                encoder_loaded_this_request=loaded,index=index_info,device=getattr(getattr(retriever,'encoder',None),'device',device),request_seconds=time.perf_counter()-started,notice=NOTICE)
