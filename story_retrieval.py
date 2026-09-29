"""Story-Level step 2: isolated fact retrieval, without consistency judgement."""
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
NOTICE='仅检索候选证据，未判断矛盾、未保存；相似度表示相对相关性，不是事实正确率或矛盾置信度。'


def fact_query(fact):
    """Include only this fact and its selected local conditions, never the story."""
    statement=' '.join(value for value in (fact['subject'],fact['predicate'],fact.get('object')) if value)
    parts=[statement,fact.get('source_text'),fact.get('context_text'),fact.get('time')]
    return '\n'.join(dict.fromkeys(value for value in parts if value))


def retrieve_facts(extraction,retriever,k=5):
    if isinstance(k,bool) or not isinstance(k,int) or k<1:
        raise ValueError('k 必须为正整数')
    if extraction['status']=='error':
        return dict(status='error',extraction=extraction,items=[],notice=NOTICE)
    items=[]
    for fact in extraction['facts']:
        query=fact_query(fact)
        start=time.perf_counter()
        try:
            evidence=retriever.search(query,k=k)
            item=dict(status='ok',fact=fact,query=query,evidence=evidence)
        except (ValueError,OSError,RuntimeError) as exc:
            item=dict(status='error',fact=fact,query=query,evidence=[],error=str(exc))
        item['retrieval_seconds']=time.perf_counter()-start
        items.append(item)
    status='partial' if extraction['status']=='partial' or any(x['status']=='error' for x in items) else 'ok'
    return dict(status=status,extraction=extraction,items=items,top_k=k,notice=NOTICE,
                retrieval_seconds=sum(x['retrieval_seconds'] for x in items))


def retrieve_story(text,device='auto',index_directory=None,k=5,judge=None,retriever=None):
    # Validate cheap options before loading either model.
    if isinstance(k,bool) or not isinstance(k,int) or k<1:
        raise ValueError('k 必须为正整数')
    from story_extraction import extract_story
    extraction=extract_story(text,device,judge=judge)
    if extraction['status']!='error' and extraction['facts'] and retriever is None:
        from encoder import Encoder
        from retrieval import Retriever
        retriever=Retriever(index_directory or ROOT/'data/index',Encoder(device=device))
    return retrieve_facts(extraction,retriever,k=k)
