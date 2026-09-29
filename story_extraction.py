"""Story-Level step 1: grounded important-fact suggestions, no retrieval or saving."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
TYPES=('character_attribute','character_knowledge','relationship','location','timeline',
       'world_rule','technology','object_state','identity','causality')
REQUIRED={'subject','predicate','type','source_ids'}
OPTIONAL={'object','context_ids','time'}
PRONOUNS={'他','她','它','他们','她们','它们','我','你','我们','此人','这个人'}
WIRE_SCHEMA={'type':'object','required':['facts'],'additionalProperties':False,
    'properties':{'facts':{'type':'array','maxItems':20,'items':{'type':'object',
    'required':sorted(REQUIRED),'additionalProperties':False,'properties':{
        **{key:{'type':'string','minLength':1} for key in ('subject','predicate')},
        'object':{'type':['string','null']},
        'type':{'type':'string','enum':list(TYPES)},
        'source_ids':{'type':'array','minItems':1,'uniqueItems':True,'items':{'type':'string'}},
        'context_ids':{'type':'array','uniqueItems':True,'items':{'type':'string'}},
        'time':{'type':['string','null']}}}}}}

def story_spans(text):
    if not isinstance(text,str) or not text.strip():raise ValueError('故事不能为空')
    spans={}
    for match in re.finditer(r'[^\r\n，,。！？!?；;]+[，,。！？!?；;]?',text):
        value=match.group()
        start=match.start()+len(value)-len(value.lstrip())
        end=match.end()-(len(value)-len(value.rstrip()))
        if start<end:spans['S'+str(len(spans)+1)]={'text':text[start:end],'start':start,'end':end}
    return spans

def _references(value,spans,required):
    if not isinstance(value,list) or (required and not value) or any(not isinstance(x,str) or x not in spans for x in value):
        raise ValueError('无效原文片段编号')
    if len(set(value))!=len(value):raise ValueError('原文片段编号重复')
    return sorted(value,key=lambda x:spans[x]['start'])

def _fact(item,text,spans,defer_grounding=False):
    if not isinstance(item,dict) or set(item)-(REQUIRED|OPTIONAL) or not REQUIRED<=set(item):
        raise ValueError('事实字段不符合 story-extraction-v2 协议')
    for key in ('subject','predicate','type'):
        if not isinstance(item[key],str) or not item[key].strip() or '\n' in item[key] or '\r' in item[key]:
            raise ValueError('事实字段必须是有效文字：'+key)
    obj=item.get('object')
    if isinstance(obj,str) and not obj.strip():obj=None
    if obj is not None and (not isinstance(obj,str) or '\n' in obj or '\r' in obj):
        raise ValueError('对象必须为文字或 null')
    if item['type'] not in TYPES:raise ValueError('事实类型无效')
    if item['subject'] in PRONOUNS or item['subject'] not in text:
        raise ValueError('主体必须明确且在故事中出现，不能猜测代词指代')
    refs=_references(item['source_ids'],spans,True)
    contexts=_references(item.get('context_ids',[]),spans,False)
    added=[]
    grounding_error=None
    if not any(item['subject'] in spans[x]['text'] for x in refs+contexts):
        antecedents=[x for x,span in spans.items() if item['subject'] in span['text']]
        first=min(spans[x]['start'] for x in refs)
        if len(antecedents)==1:
            antecedent=antecedents[0]
            end=spans[antecedent]['end']
            if end<=first and not re.search(r'[。！？!?；;\r\n]',text[end-1:first]):
                contexts=_references(contexts+[antecedent],spans,False)
                added=[antecedent]
        if not added:
            grounding_error='引用缺少主体依据，请选择明确先行词的 context_ids'
            if not defer_grounding:raise ValueError(grounding_error)
    event_time=item.get('time')
    if event_time is not None and (not isinstance(event_time,str) or not event_time.strip() or event_time not in text):
        raise ValueError('时间必须来自原文或为 null')
    start=min(spans[x]['start'] for x in refs);end=max(spans[x]['end'] for x in refs)
    fact=dict(**{k:item[k] for k in ('subject','predicate','type')},object=obj,
        source_text=text[start:end],source_start=start,source_end=end,source_ids=refs,
        grounding_added_context_ids=added,context_ids=contexts,context_text='\n'.join(spans[x]['text'] for x in contexts),time=event_time)
    if grounding_error:fact['grounding_error']=grounding_error
    return fact

def decode_facts(value,text,spans,defer_grounding=False):
    if not isinstance(value,dict) or set(value)!={'facts'} or not isinstance(value['facts'],list) or len(value['facts'])>20:
        raise ValueError('输出必须为包含 0 到 20 条 facts 的 JSON 对象')
    facts=[];rejected=[];duplicates=[];used=set()
    for index,item in enumerate(value['facts'],1):
        try:
            fact=_fact(item,text,spans,defer_grounding=defer_grounding)
            if any({k:v for k,v in old.items() if k!='id'}==fact for old in facts):
                duplicates.append(index);continue
            fact['id']='F'+str(index);facts.append(fact)
            used.update(fact['source_ids']);used.update(fact['context_ids'])
        except ValueError as exc:rejected.append(dict(id='F'+str(index),fact_index=index,error=str(exc),raw_fact=item))
    status='partial' if rejected and facts else 'error' if rejected else 'ok'
    return dict(status=status,facts=facts,rejected=rejected,duplicates=duplicates,
                unselected_spans={k:v for k,v in spans.items() if k not in used})

def apply_importance_review(result,value,spans,recover=False):
    ids={fact['id'] for fact in result['facts']}
    if not isinstance(value,dict) or set(value)!={'keep','discard'} or not isinstance(value['keep'],list) or not isinstance(value['discard'],list):
        raise ValueError('重要性审查字段无效')
    kept=value['keep'];dropped=value['discard']
    if any(not isinstance(x,str) for x in kept) or any(not isinstance(x,dict) or set(x)!={'id','reason'} or not isinstance(x['id'],str) or not isinstance(x['reason'],str) or not x['reason'].strip() for x in dropped):
        raise ValueError('重要性审查编号/原因无效')
    decisions=kept+[x['id'] for x in dropped]
    problems=len(set(decisions))!=len(decisions) or set(decisions)!=ids
    if problems and not recover:
        raise ValueError('重要性审查必须不重不漏覆盖全部候选')
    issues=[]
    pending=[]
    for fact in result['facts']:
        count=decisions.count(fact['id'])
        if count!=1:
            issues.append(dict(id=fact['id'],error='未提供筛选决定' if count==0 else '筛选决定重复或冲突'))
            pending.append(fact)
    for label in sorted(set(decisions)-ids):
        issues.append(dict(id=label,error='未知候选编号'))
    uncertain={fact['id'] for fact in pending}
    facts=[fact for fact in result['facts'] if fact['id'] in kept and fact['id'] not in uncertain]
    discarded=[row for row in dropped if row['id'] in ids and row['id'] not in uncertain]
    used={ref for fact in facts for ref in fact['source_ids']+fact['context_ids']}
    return dict(result,status='partial' if issues else result['status'],facts=facts,
                candidates=result['facts'],discarded=discarded,pending_facts=pending,review_issues=issues,
                unselected_spans={key:span for key,span in spans.items() if key not in used})

def decode_extraction_reply(raw):
    """Accept a complete leading object plus explicit prose, never repair JSON."""
    stripped=raw.strip()
    value,end=json.JSONDecoder().raw_decode(stripped)
    tail=stripped[end:].strip()
    if not tail:return value,[]
    if tail=='"':return value,['single_trailing_quote_removed']
    if not re.match(r'^(说明|解释)\s*[：:]',tail) or any(c in tail for c in '{}[]'):
        raise ValueError('提取回复包含额外内容或多个JSON对象')
    return value,['trailing_explanation_removed']


def extraction_budget(judge,messages):
    """Reserve output inside the existing 4096-token local inference budget."""
    tokenizer=getattr(judge,'tokenizer',None)
    if tokenizer is None:return dict(input_tokens=None,max_new_tokens=1536,total_limit=4096)
    count=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    available=4096-count
    if available<256:
        raise ValueError(f'故事提取请求占{count} tokens，4096预算不足以预留256输出tokens；未截断原文')
    return dict(input_tokens=count,max_new_tokens=min(1536,available),total_limit=4096)


def extract_story(text,device='auto',judge=None):
    spans=story_spans(text)
    if len(text)>800:raise ValueError('本步骤只支持不超过800字符的短故事，不会截断')
    prompt=(ROOT/'prompts/story_extraction_v2.txt').read_text(encoding='utf-8')
    prompt+='\nsource_spans是编号到原文文字的引用表。只输出一个JSON对象，结束后立即停止，不输出说明、解释或Markdown。'
    messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(
        dict(story=text,source_spans={key:span['text'] for key,span in spans.items()}),ensure_ascii=False,separators=(',',':'))}]
    if judge is None:
        from qwen_judge import QwenJudge
        judge=QwenJudge(device)
    raw=''
    review_raw=''
    timings=[]
    candidates=[]
    result=dict(status='error',facts=[],rejected=[],duplicates=[])
    stage='candidate_extraction'
    budget=None
    normalizations=[]
    try:
        budget=extraction_budget(judge,messages)
        raw=judge._generate(messages,max_new_tokens=budget['max_new_tokens'])
        timings.append(dict(stage='candidate_extraction',**judge.last_generation))
        value,normalizations=decode_extraction_reply(raw)
        result=decode_facts(value,text,spans,defer_grounding=True)
        candidates=result['facts']
        facts=[];discarded=[];pending=[];issues=[];review_outputs={}
        review_prompt=(ROOT/'prompts/story_importance_v2.txt').read_text(encoding='utf-8')
        for candidate in candidates:
            stage='importance_review_'+candidate['id']
            response=''
            try:
                public={k:v for k,v in candidate.items() if k!='grounding_error'}
                response=judge._generate([{'role':'system','content':review_prompt},
                    {'role':'user','content':json.dumps(dict(candidate=public),ensure_ascii=False)}],max_new_tokens=160)
                decision=json.loads(response)
                if (not isinstance(decision,dict) or set(decision)!={'decision','reason'}
                    or decision['decision'] not in ('keep','discard')
                    or not isinstance(decision['reason'],str) or not decision['reason'].strip()):
                    raise ValueError('逐条重要性审查必须提供唯一 decision 和非空 reason')
                if decision['decision']=='discard':
                    discarded.append(dict(id=candidate['id'],reason=decision['reason']))
                elif candidate.get('grounding_error'):
                    result['rejected'].append(dict(id=candidate['id'],error=candidate['grounding_error'],raw_fact=candidate))
                else:
                    facts.append(candidate)
            except ValueError as exc:
                pending.append(candidate)
                issues.append(dict(id=candidate['id'],error=str(exc)))
                response=getattr(exc,'raw_output',response)
            finally:
                review_outputs[candidate['id']]=response
                timings.append(dict(stage=stage,**getattr(judge,'last_generation',{})))
        review_raw=json.dumps(review_outputs,ensure_ascii=False)
        used={ref for fact in facts for ref in fact['source_ids']+fact['context_ids']}
        status='partial' if issues or result['rejected'] else 'ok'
        result=dict(result,status=status,facts=facts,candidates=candidates,discarded=discarded,
            pending_facts=pending,review_issues=issues,
            unselected_spans={key:span for key,span in spans.items() if key not in used})
    except ValueError as exc:
        result=dict(result,status='error',facts=[],error=str(exc))
        last=dict(getattr(judge,'last_generation',{}))
        if last and not any(t['stage']==stage for t in timings):
            timings.append(dict(stage=stage,**last))
        if candidates:
            result['candidates']=candidates
            review_raw=getattr(exc,'raw_output',review_raw)
        else:
            raw=getattr(exc,'raw_output',raw)
    return dict(result,story=text,raw_output=raw,prompt_version='story-extraction-v2',
                transport_version='compact-spans-v1',generation_budget=budget,format_normalizations=normalizations,
                timing=dict(seconds=sum(t.get('seconds',0) for t in timings),calls=timings),
                importance_raw_output=review_raw,importance_prompt_version='story-importance-v2',device=judge.device,
                notice='只筛选事实建议；未检索、未判断矛盾、未保存。原文引用有效不保证事实理解正确。')
