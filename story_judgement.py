"""Story-Level step 3: evidence-bound per-fact consistency judgement."""
import json
from pathlib import Path
import time
from story_retrieval import retrieve_story

ROOT=Path(__file__).resolve().parent
NOTICE='本地逐事实一致性建议，未保存。判断可能出错；检索相似度不是事实正确率或矛盾置信度。'


def validate_answer(value,evidence):
    if not isinstance(value,dict) or set(value)!={'verdict','reason','evidence_ids'}:
        raise ValueError('判断必须包含 verdict、reason、evidence_ids')
    if value['verdict'] not in ('一致','矛盾','不确定'):
        raise ValueError('判断只能为一致、矛盾或不确定')
    if not isinstance(value['reason'],str) or not value['reason'].strip():
        raise ValueError('判断原因不能为空')
    ids=value['evidence_ids']
    available={row['id']:row for row in evidence}
    if not isinstance(ids,list) or any(not isinstance(id,str) or id not in available for id in ids) or len(set(ids))!=len(ids):
        raise ValueError('证据引用无效、重复或不属于当前事实检索结果')
    if value['verdict']!='不确定' and not ids:
        raise ValueError('一致或矛盾判断必须引用证据')
    return dict(status='ok',**value,evidence=[available[id] for id in ids])


def decode_reply(raw,evidence):
    normalizations=[]
    try:
        value=json.loads(raw)
    except json.JSONDecodeError:
        # One observed EOS artefact. Empty citations are legal only for uncertainty.
        stripped=raw.strip()
        if not stripped.endswith('"evidence_ids"}'):raise
        value=json.loads(stripped[:-1]+':[]}')
        if not isinstance(value,dict) or value.get('verdict')!='不确定':raise
        normalizations=['missing_empty_uncertain_evidence_array']
    return dict(validate_answer(value,evidence),format_normalizations=normalizations)


def check_facts(retrieval,judge):
    prompt=(ROOT/'prompts/story_consistency_v2.txt').read_text(encoding='utf-8')
    items=[];calls=[]
    for item in retrieval['items']:
        raw='';attempts=[];started=time.perf_counter()
        if item['status']!='ok':
            answer=dict(status='error',verdict=None,reason='',evidence_ids=[],evidence=[],error='未判断：'+item.get('error','检索失败'))
        elif not item['evidence']:
            answer=dict(status='ok',verdict='不确定',reason='没有检索到候选证据，无法核对。',evidence_ids=[],evidence=[])
        else:
            # No story, neighbouring facts, retrieval query or similarity ranking as truth.
            evidence=[{k:row[k] for k in ('id','text','file','start_line','end_line','heading_path') if k in row} for row in item['evidence']]
            fact={k:item['fact'][k] for k in ('id','subject','predicate','object','source_text','context_text','time') if k in item['fact']}
            for attempt in (1,2):
                raw=''
                messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(dict(fact=fact,evidence=evidence),ensure_ascii=False))]
                if attempt==2:
                    messages[0]['content']+='\n上次回复格式或引用协议不合格。重新核对相同事实与证据，只输出完整JSON，三个字段不能缺省，reason只写一句话。'
                try:
                    raw=judge._generate(messages,max_new_tokens=256)
                    answer=decode_reply(raw,item['evidence'])
                    attempts.append(dict(attempt=attempt,status='ok',raw_output=raw,format_normalizations=answer['format_normalizations']))
                    break
                except (ValueError,OSError,RuntimeError) as exc:
                    raw=getattr(exc,'raw_output',raw)
                    answer=dict(status='error',verdict=None,reason='',evidence_ids=[],evidence=[],error=str(exc))
                    attempts.append(dict(attempt=attempt,status='error',raw_output=raw,error=str(exc)))
                    if not isinstance(exc,ValueError):break
                finally:
                    calls.append(dict(fact_id=item['fact']['id'],attempt=attempt,**getattr(judge,'last_generation',{})))
        answer.update(raw_output=raw,attempts=attempts,wall_seconds=time.perf_counter()-started)
        items.append(dict(item,judgement=answer))
    errors=sum(x['judgement']['status']=='error' for x in items)
    counts={verdict:sum(x['judgement']['verdict']==verdict for x in items) for verdict in ('一致','矛盾','不确定')}
    status='error' if retrieval['status']=='error' else 'partial' if retrieval['status']=='partial' or errors else 'ok'
    if counts['矛盾']:verdict='矛盾'
    elif status!='ok' or counts['不确定']:verdict='不确定'
    elif items:verdict='一致'
    else:verdict=None
    complete=status=='ok'
    display=(verdict or '未发现可检查的重要事实') if complete else '检查未完成'+('；已发现矛盾' if counts['矛盾'] else '，不能给出完整结论')
    return dict(retrieval,status=status,items=items,notice=NOTICE,prompt_version='story-consistency-v2',judgement_calls=calls,
                summary=dict(verdict=verdict,counts=counts,errors=errors,checked_facts=len(items),
                    complete=complete,display=display,
                    scope='仅已提取并检索的事实；不证明整段故事无矛盾。'))


def check_story(text,device='auto',index_directory=None,k=5,judge=None,retriever=None):
    from story_extraction import story_spans
    story_spans(text)
    if len(text)>800:raise ValueError('本步骤只支持不超过800字符的短故事，不会截断')
    if isinstance(k,bool) or not isinstance(k,int) or k<1:raise ValueError('k 必须为正整数')
    if judge is None:
        from qwen_judge import QwenJudge
        judge=QwenJudge(device)
    started=time.perf_counter()
    retrieval=retrieve_story(text,device,index_directory,k,judge=judge,retriever=retriever)
    result=check_facts(retrieval,judge)
    return dict(result,pipeline_seconds=time.perf_counter()-started,model_load_seconds=getattr(judge,'load_seconds',None))
