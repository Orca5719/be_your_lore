"""Bounded JSON transport using the existing pinned offline model loader."""
import json


def call_json(llm,messages,max_output=1536,validator=None):
    attempts=[]
    for attempt in range(2):
        current=list(messages)
        if attempt:
            current=current+[{'role':'user','content':'上次回复格式不合格：'+attempts[-1]['error']+'。重新只返回符合要求的一个完整JSON对象，不加说明。'}]
        raw=''
        llm.last_generation={}
        try:
            tokenizer=getattr(llm,'tokenizer',None)
            count=len(tokenizer.apply_chat_template(current,tokenize=True,add_generation_prompt=True)) if tokenizer else None
            budget=min(max_output,4096-count) if count is not None else max_output
            if budget<256:
                raise RuntimeError(f'请求输入{count} tokens，剩余预算不足256；未截断原文')
            raw=llm._generate(current,max_new_tokens=budget)
            value=json.loads(raw)
            if not isinstance(value,dict):raise ValueError('输出必须为JSON对象')
            if validator is not None:validator(value)
            attempts.append(dict(attempt=attempt+1,status='ok',raw_output=raw,input_tokens=count,max_new_tokens=budget,timing=dict(llm.last_generation)))
            return value,dict(status='ok',attempts=attempts)
        except (ValueError,RuntimeError,OSError) as exc:
            attempts.append(dict(attempt=attempt+1,status='error',raw_output=getattr(exc,'raw_output',raw),error=str(exc),error_type=type(exc).__name__,timing=dict(llm.last_generation)))
            if not isinstance(exc,ValueError):break
    return None,dict(status='error',attempts=attempts)
