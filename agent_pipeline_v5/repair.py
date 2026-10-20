"""5C1: two observed field-name typos, without semantic inference."""
from copy import deepcopy
import json

POLICY = 'schema_aliases_v1'


def repair_payload(value):
    result=deepcopy(value)
    actions=[]
    if not isinstance(result,dict):
        return result,actions
    for collection,old,new in (('events','mentalological_state','mental_state'),
                               ('ignored_spans','source_Ids','source_id')):
        rows=result.get(collection)
        if not isinstance(rows,list):
            continue
        for index,row in enumerate(rows):
            if isinstance(row,dict) and old in row and new not in row:
                row[new]=row.pop(old)
                actions.append({'path':f'{collection}[{index}]','from':old,'to':new})
    return result,actions


def repair_raw(raw):
    try:
        value=json.loads(raw)
    except (ValueError,TypeError):
        return raw,[]
    result,actions=repair_payload(value)
    # An untouched reply must retain byte-for-byte wire text.
    return (json.dumps(result,ensure_ascii=False,separators=(',',':')),actions) if actions else (raw,[])


def replay_archive(directory, output):
    from .repair_audit import replay_archive as replay
    return replay(directory,output)
