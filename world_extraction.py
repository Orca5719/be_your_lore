"""Compact extraction wire format; source text is reconstructed, never generated."""
import re
from world_records import validate_plan

KEYS={'k':'kind','e':'entity','y':'entity_type','c':'category','x':'target','t':'time','f':'valid_from','u':'valid_until'}

def source_spans(text):
    # Keep every non-delimiter character, including conjunctions and omitted subjects.
    parts=[part.strip() for part in re.split(r'[，,。！？!?；;\r\n]+',text) if part.strip()]
    return {'S'+str(i):part for i,part in enumerate(parts,1)}

MISSING_TIME={'未知','未指定','未提供','不明确','无','null','None'}

def _decode_record(item,text,spans):
    if not isinstance(item,dict) or set(item)-set(KEYS)-{'s'}:
        raise ValueError('归档短字段无效')
    record={KEYS[key]:val for key,val in item.items() if key in KEYS}
    # Normalize missingness only at the model boundary, never stored data.
    for key in ('time','valid_from','valid_until'):
        val=record.get(key)
        if isinstance(val,str) and (not val.strip() or (val.strip() in MISSING_TIME and val not in text)):
            record[key]=None
    kind=record.get('kind')
    if kind in ('entity','timepoint'):
        if 's' in item:
            raise ValueError('实体/时间节点原文由名称生成，不接受片段编号')
        record['text']=record.get('entity')
        if kind=='entity':
            record.setdefault('category',record.get('entity_type'))
        else:
            record.setdefault('category','时间节点')
            record.setdefault('time',record.get('entity'))
    else:
        ref=item.get('s')
        if not isinstance(ref,str) or ref not in spans:
            raise ValueError('无效原文片段编号：'+str(ref))
        record['text']=spans[ref]
    return validate_plan({'records':[record]},text)[0]

def decode_records(value,text,spans,recover=False):
    if not isinstance(text,str) or not text.strip():
        raise ValueError('输入不能为空')
    if not isinstance(value,dict) or set(value)!={'records'} or not isinstance(value['records'],list) or not 1<=len(value['records'])<=20:
        raise ValueError('归档输出必须包含 1 到 20 个 records')
    records=[]
    rejected=[]
    duplicates=[]
    for index,item in enumerate(value['records'],1):
        try:
            record=_decode_record(item,text,spans)
            if record in records:
                if not recover:
                    raise ValueError('重复的通用条目')
                duplicates.append(index)
                continue
            records.append(record)
        except ValueError as exc:
            if not recover:
                raise
            rejected.append(dict(record_index=index,error=str(exc),raw_record=item))
    if not records:
        raise ValueError('所有归档条目均未通过校验：'+'；'.join(row['error'] for row in rejected))
    if recover:
        return dict(records=records,rejected=rejected,duplicates=duplicates,
                    status='partial' if rejected else 'complete',protocol_version='world-extraction-decoder-v2')
    return records
