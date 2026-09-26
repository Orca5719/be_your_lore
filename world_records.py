"""General world records: typed suggestions, source spans, time scope and batch storage."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import uuid
from entries import atomic_write

KINDS={'entity','attribute','relation','rule','event','timepoint'}
FIELDS=('kind','entity','entity_type','category','text','target','time','valid_from','valid_until')
PRONOUNS={'他','她','它','他们','她们','它们','我','你','我们','这个人','此人'}
HEADER='# 用户确认的通用世界观条目（由 world_records.json 生成，请勿直接编辑）'

def validate_plan(value, source):
    if not isinstance(source,str) or not source.strip():
        raise ValueError('输入不能为空')
    if not isinstance(value,dict) or set(value)!={'records'} or not isinstance(value['records'],list) or not 1<=len(value['records'])<=20:
        raise ValueError('计划必须包含 1 到 20 个 records；复杂输入请拆分')
    result=[]
    signatures=set()
    for item in value['records']:
        if not isinstance(item,dict) or set(item)-set(FIELDS) or not {'kind','category','text'}<=set(item):
            raise ValueError('通用条目字段无效')
        record={key:item.get(key) for key in FIELDS}
        if not isinstance(record['kind'],str) or record['kind'] not in KINDS:
            raise ValueError('条目类型无效')
        for key in FIELDS[1:]:
            val=record[key]
            if val is not None and (not isinstance(val,str) or not val.strip() or (key!='text' and ('\n' in val or '\r' in val))):
                raise ValueError('条目字段必须是有效文字：'+key)
        if not record['category'] or not record['text'] or record['text'] not in source:
            raise ValueError('条目原文必须逐字来自输入，类别不能为空')
        for key in ('entity','target','time','valid_from','valid_until'):
            if record[key] is not None and record[key] not in source:
                raise ValueError('名称或时间不在原输入中：'+key)
        if record['entity'] in PRONOUNS or record['target'] in PRONOUNS:
            raise ValueError('主体/对象是代词，请补充真实名称后重试，未保存')
        if record['kind'] in ('entity','attribute','relation','timepoint') and record['entity'] is None:
            raise ValueError('该条目需要明确实体')
        if record['kind']=='entity' and not record['entity_type']:
            raise ValueError('实体需建议实体类型，允许自定义类型')
        if record['kind']=='relation' and not record['target']:
            raise ValueError('关系需明确对象')
        if record['kind']=='timepoint' and not record['time']:
            raise ValueError('时间节点需明确原文时间标签')
        signature=tuple(record[key] for key in FIELDS)
        if signature in signatures:
            raise ValueError('重复的通用条目')
        signatures.add(signature)
        result.append(record)
    return result


def build_catalog(chunks, records):
    rows=[]
    names=set()
    times=set()
    for chunk in chunks:
        headings=chunk.get('heading_path',[])
        if len(headings)>=2:
            label=headings[-2]
            if re.fullmatch(r'(?:[0-9]{3,4}年(?:[0-9]{1,2}月(?:[0-9]{1,2}日)?)?|第[零〇一二三四五六七八九十百千万0-9]+[话章节卷幕])',label):
                times.add(label)
            else:
                names.add(label)
        rows.append(dict(text=chunk['text'],source={key:chunk.get(key) for key in ('id','file','start_line','end_line')},entity=headings[-2] if len(headings)>=2 else None,category=headings[-1] if headings else None))
    for record in records:
        for key in ('entity','target'):
            if record.get(key):
                names.add(record[key])
        for key in ('time','valid_from','valid_until','event_time'):
            if record.get(key):
                times.add(record[key])
        rows.append(dict(text=record['text'],source={'id':record.get('id'),'file':record.get('file','structured_store')},entity=record.get('entity'),category=record.get('category'),record=record))
    return dict(names=sorted(names),times=sorted(times),rows=rows)


def presence(label, catalog):
    matches=[row for row in catalog['rows'] if label and (label==row.get('entity') or label in row['text'] or label in [row.get('record',{}).get(key) for key in ('target','time','valid_from','valid_until','event_time')])]
    return dict(status='mentioned' if matches else 'not_found_in_catalog',message='完整资料中已提及，是否为同一实体/节点仍需核对' if matches else '完整资料未找到这个精确名称/时间标签；可能是新增或别名，待作者确认',sources=[row['source'] for row in matches])


class WorldStore:
    def __init__(self,path):
        self.path=Path(path)

    def read(self):
        if not self.path.exists():
            return {'schema_version':1,'records':[]},'empty'
        raw=self.path.read_bytes()
        try:
            value=json.loads(raw)
        except (ValueError,UnicodeError) as exc:
            raise ValueError('通用条目库损坏，拒绝覆盖') from exc
        if not isinstance(value,dict) or set(value)!={'schema_version','records'} or value['schema_version']!=1 or not isinstance(value['records'],list):
            raise ValueError('通用条目库格式不匹配')
        seen=set()
        for record in value['records']:
            if not isinstance(record,dict) or set(record)!=set(FIELDS)|{'id','created_at','source_input'}:
                raise ValueError('已保存通用条目损坏')
            validate_plan({'records':[{key:record[key] for key in FIELDS}]},record['source_input'])
            if not isinstance(record['id'],str) or record['id'] in seen or not isinstance(record['created_at'],str):
                raise ValueError('条目 ID/时间无效')
            seen.add(record['id'])
        return value,hashlib.sha256(raw).hexdigest()

    def preview(self,records,source,catalog):
        records=validate_plan({'records':records},source)
        _,revision=self.read()
        items=[]
        for record in records:
            items.append(dict(proposed=record,existing=[row for row in catalog['rows'] if row.get('entity')==record['entity'] and row.get('category')==record['category']],
                entity_presence=presence(record['entity'],catalog) if record['entity'] else None,
                target_presence=presence(record['target'],catalog) if record['target'] else None,
                time_presence=presence(record['time'],catalog) if record['time'] else None))
        return dict(status='pending_confirmation',store_revision=revision,source_input=source,items=items)

    def commit(self,preview,selected=None):
        indices=list(range(len(preview['items']))) if selected is None else selected
        if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=len(preview['items']) for i in indices):
            raise ValueError('保存选择无效')
        proposals=validate_plan({'records':[preview['items'][i]['proposed'] for i in indices]},preview['source_input'])
        self.path.parent.mkdir(parents=True,exist_ok=True)
        lock=self.path.with_suffix('.json.lock')
        try:
            descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError('通用资料库正在保存，请稍后重试') from exc
        try:
            os.close(descriptor)
            value,revision=self.read()
            if revision!=preview['store_revision']:
                raise ValueError('预览后通用资料已改变，请重新预览')
            def signature(record):
                return tuple(record.get(key) for key in sorted(FIELDS))
            seen={signature(record) for record in value['records']}
            saved=[]
            for record in proposals:
                sig=signature(record)
                if sig in seen:
                    raise ValueError('选中条目含重复内容，本批未保存')
                seen.add(sig)
                saved.append(dict(record,id=uuid.uuid4().hex,created_at=datetime.now(timezone.utc).isoformat(),source_input=preview['source_input']))
            value['records']+=saved
            atomic_write(self.path,json.dumps(value,ensure_ascii=False,indent=2))
            return saved
        finally:
            lock.unlink(missing_ok=True)

    def export(self,path):
        path=Path(path)
        if path.exists() and not path.read_text(encoding='utf-8').startswith(HEADER):
            raise ValueError('目标已有手写资料，拒绝覆盖')
        value,_=self.read()
        lines=[HEADER,'']
        for record in value['records']:
            lines+=['## '+(record['entity'] or '世界规则'),'### '+record['category'],'']
            for label,key in [('条目类型','kind'),('实体类型','entity_type'),('关系对象','target'),('事件时间','time'),('生效起点','valid_from'),('生效终点','valid_until')]:
                if record[key]:
                    lines+=['    '+label+'：'+record[key]]
            lines+=['    主体：'+record['entity']] if record['entity'] else []
            lines+=['    '+line for line in record['text'].splitlines()]
            lines+=['']
        atomic_write(path,'\n'.join(lines))




def validate_world_verdict(raw,record,results):
    from judgment_protocol import validate_answer
    try:
        value=json.loads(raw)
    except (ValueError,TypeError) as exc:
        raise ValueError('逐条判断不是有效 JSON') from exc
    if not isinstance(value,dict) or set(value)!={'verdict','evidence','reason'} or not isinstance(value['evidence'],list):
        raise ValueError('逐条判断字段无效')
    aliases={'E'+str(i):row for i,row in enumerate(results,1)}
    citations=[]
    seen=set()
    for label in value['evidence']:
        if not isinstance(label,str) or label not in aliases or label in seen:
            raise ValueError('证据编号无效或重复')
        seen.add(label)
        row=aliases[label]
        citations.append({'chunk_id':row['id'],'quote':row['text']})
    finding={'input_quote':record['text'],'verdict':value['verdict'],'evidence':citations,'reason':value['reason'],'new_candidates':[]}
    return validate_answer(json.dumps({'findings':[finding]},ensure_ascii=False),record['text'],{row['id']:row['text'] for row in results})

