import copy
import json
import pytest
from agent_pipeline_v5.wire import expand_event, expand_payload
from agent_pipeline_v5.extractor import WireBatchModel


def compact(**changes):
    value=dict(actors=['雷'],event='雷没有受伤',source_ids=['S1'],modality='observed',check_reason='mechanism')
    value.update(changes)
    return value

SPANS={'S1':{'text':'雷没有受伤。'},'S2':{'text':'雷说自己受伤了。'}}

@pytest.mark.parametrize('modality',['observed','speech','belief','plan','dream','conditional','inferred'])
def test_expand_preserves_semantics_and_derives_explicit(modality):
    row=compact(modality=modality,conditions=['如果结界打开'],mental_state='怀疑',context_ids=['S2'])
    original=copy.deepcopy(row)
    event=expand_event(row,'E3',SPANS)
    assert event['explicit'] is (modality!='inferred')
    assert event['conditions']==row['conditions'] and event['mental_state']=='怀疑'
    assert event['context_ids']==['S2'] and event['event']=='雷没有受伤'
    assert row==original


def test_optional_omission_means_only_empty_defaults():
    event=expand_event(compact(),'E1',SPANS)
    assert event['conditions']==event['context_ids']==[] and event['mental_state'] is None


@pytest.mark.parametrize('changes',[{'source_ids':['S99']},{'context_ids':['S1']},{'actors':[]},
    {'modality':'invalid'},{'explicit':False},{'unknown':'x'},{'conditions':None},{'modality':None}])
def test_invalid_compact_events_rejected(changes):
    with pytest.raises(ValueError):expand_event(compact(**changes),'E1',SPANS)


def test_three_exit_envelope_and_full_wire_compatibility():
    value={'events':[compact()], 'ignored_spans':[{'source_id':'S2','reason':'routine'}],'non_event_span_ids':[]}
    out=expand_payload(value)
    assert out['ignored_spans']==value['ignored_spans'] and 'id' not in out['events'][0]
    assert out['events'][0]['explicit'] is True
    with pytest.raises(ValueError):expand_payload({**value,'extra':[]})


class FakeBatchModel:
    device='cpu'
    load_seconds=0
    enabled=False
    def __init__(self,raw):self.raw=raw;self.seen=[];self.last_batch_generation={}
    def _generate_batch(self,batches,max_new_tokens):
        self.seen.append((copy.deepcopy(batches),max_new_tokens))
        self.last_batch_generation={'call_id':'CALL1','truncated_indices':[]}
        return [self.raw]


def test_wire_adapter_keeps_raw_and_replaces_only_system_prompt():
    raw=json.dumps({'events':[compact()],'ignored_spans':[],'non_event_span_ids':[]})
    delegate=FakeBatchModel(raw)
    model=WireBatchModel(delegate,'compact')
    messages=[{'role':'system','content':model.original_prompt},{'role':'user','content':'unchanged'}, {'role':'user','content':'repair'}]
    saved=copy.deepcopy(messages)
    result=model._generate_batch([messages],1536)
    assert messages==saved and delegate.seen[0][0][0][1:]==messages[1:]
    assert delegate.seen[0][1]==1536
    assert delegate.seen[0][0][0][0]['content']==model.compact_prompt
    assert json.loads(result[0])['events'][0]['explicit']
    assert model.wire_records[0]['raw_output']==raw
    assert model.wire_records[0]['expanded_output']==result[0]


def test_malformed_or_truncated_wire_is_not_accepted_as_empty():
    for raw in ['{bad',json.dumps({'events':[compact(unknown=1)],'ignored_spans':[],'non_event_span_ids':[]})]:
        delegate=FakeBatchModel(raw); model=WireBatchModel(delegate,'compact')
        out=model._generate_batch([[{'role':'system','content':model.original_prompt}]],1536)[0]
        assert model.wire_records[0]['error']
        assert json.loads(out).get('events') is None


def test_original_adapter_does_not_rewrite_output():
    delegate=FakeBatchModel('unchanged');model=WireBatchModel(delegate,'original')
    assert model._generate_batch([[{'role':'system','content':model.original_prompt}]],1536)==['unchanged']


def test_compact_preserves_cross_sentence_actor_and_condition():
    from agent_pipeline_v2.extractor import _decode_window
    value={'events':[compact(event='他在结界打开后才会受伤',source_ids=['S2'],context_ids=['S1'],conditions=['结界打开后'])],
           'ignored_spans':[],'non_event_span_ids':[]}
    spans={'S1':{'text':'雷站在那里。'},'S2':{'text':'他在结界打开后才会受伤。'}}
    events,_,_=_decode_window(expand_payload(value),spans,{'target_ids':['S2'],'context_ids':['S1']},1)
    assert events[0]['actors']==['雷'] and events[0]['conditions']==['结界打开后']
    assert events[0]['context_ids']==['S1']


def test_existing_nonsemantic_aliases_still_work():
    row=compact();row['checkReason']=row.pop('check_reason')
    assert expand_event(row,'E1',SPANS)['check_reason']=='mechanism'


def test_generation_resource_stop_is_sticky():
    from agent_pipeline_v5.resources import ResourceLimit
    calls=[]
    def guard():
        calls.append(1)
        raise ResourceLimit('disk full')
    delegate=FakeBatchModel('unused');model=WireBatchModel(delegate,'compact',guard=guard)
    messages=[[{'role':'system','content':model.original_prompt}]]
    for _ in range(2):
        with pytest.raises(RuntimeError):model._generate_batch(messages,1536)
    assert calls==[1] and delegate.seen==[]
