import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_pipeline_v5.extractor import WireBatchModel
from agent_pipeline_v5.runner import run_extraction_arm, ensure_manifest


ROOT=Path(__file__).resolve().parents[1]


class RecordingModel:
    device='cpu';load_seconds=0;enabled=False
    def __init__(self,replies):
        self.replies=replies;self.seen=[];self.last_batch_generation={}
    def _generate_batch(self,batches,max_new_tokens=1536):
        self.seen.append((copy.deepcopy(batches),max_new_tokens))
        n=len(self.seen)
        self.last_batch_generation=dict(call_id=f'L{n}',profile=dict(call_id=f'L{n}',status='ok',input_tokens=20,output_tokens=10,total_time=.1,prefill_time=.01,decode_time=.09,batch_size=1),input_tokens=[20],generated_tokens=[10],seconds=.1,truncated_indices=[])
        return [self.replies[min(n-1,len(self.replies)-1)]]
    def clear_traces(self):pass


def event(**changes):
    result=dict(actors=['甲'],event='甲没有受伤',mental_state=None,explicit=True,modality='observed',conditions=[],source_ids=['S1'],context_ids=[],check_reason='state_time_space')
    result.update(changes);return result


def payload(events=(),ignored=()):
    return json.dumps(dict(events=list(events),ignored_spans=list(ignored),non_event_span_ids=[]),ensure_ascii=False)


def test_semantic_changes_only_system_keeps_output_and_logs_prompt_digest():
    raw=payload([event()]);delegate=RecordingModel([raw])
    adapter=WireBatchModel(delegate,'original',prompt_policy='semantic_v1')
    messages=[{'role':'system','content':adapter.original_prompt},{'role':'user','content':'{"target_spans":{"S1":"甲没有受伤。"}}'},{'role':'user','content':'原重试消息'}]
    before=copy.deepcopy(messages)
    assert adapter._generate_batch([messages],1536)==[raw]
    sent=delegate.seen[0][0][0]
    assert messages==before and sent[1:]==messages[1:] and delegate.seen[0][1]==1536
    actual=(ROOT/'agent_pipeline_v5/prompts/extractor_semantic_v1.txt').read_text(encoding='utf-8')
    assert sent[0]['content']==actual and actual!=adapter.original_prompt
    record=adapter.wire_records[0]
    assert record['raw_output']==record['expanded_output']==raw and record['schema_repairs']==[]
    assert record['prompt_policy']=='semantic_v1'
    assert record['system_prompt_sha256']==hashlib.sha256(actual.encode('utf-8')).hexdigest()


def test_semantic_format_example_passes_frozen_schema_without_story_facts():
    from agent_pipeline_v2.extractor import _decode_window
    prompt=(ROOT/'agent_pipeline_v5/prompts/extractor_semantic_v1.txt').read_text(encoding='utf-8')
    examples=[json.loads(line) for line in prompt.splitlines() if line.startswith('{"events":')]
    assert len(examples)==1, 'need a complete JSON format example, not prose alone'
    example=examples[0]
    # Placeholders are schema examples, not facts about the regression stories.
    row=example['events'][0]
    assert row['actors']==['<原文主体>'] and row['event']=='<原文直接支持的完整命题>'
    spans={'S1':{'text':'<原文主体>：<原文直接支持的完整命题>。'},'S2':{'text':'普通过程。'},'S3':{'text':'随后，'}}
    events,ignored,non_events=_decode_window(example,spans,{'target_ids':['S1','S2','S3'],'context_ids':[]},1)
    assert events[0]['check_reason'] in {'mechanism','knowledge_relation','state_time_space','consequence_support'}
    assert ignored==[{'source_id':'S2','reason':'routine'}] and non_events==['S3']


def test_missing_check_reason_still_fails_instead_of_inventing_reason(tmp_path):
    broken=event();del broken['check_reason']
    model=RecordingModel([payload([broken])])
    result=run_extraction_arm({'arm':'semantic_plain','warmup':False},[{'id':'C1','story':'甲没有受伤。'}],model,tmp_path)
    assert not result['complete'] and len(model.seen)==2
    ex=result['rows'][0]['result']['benchmark_stages']['extraction']
    assert ex['events']==[] and ex['uncovered_span_ids']==['S1']
    assert all('check_reason' not in json.loads(a['raw_output'])['events'][0] for a in ex['calls'][0]['attempts'])


@pytest.mark.parametrize('protocol,repair',[('compact','none'),('original','schema_aliases_v1')])
def test_semantic_cannot_silently_combine_unaccepted_strategies(protocol,repair):
    with pytest.raises(ValueError):WireBatchModel(RecordingModel(['unused']),protocol,prompt_policy='semantic_v1',repair_policy=repair)


def test_semantic_retry_keeps_prompt_and_original_validation_budget(tmp_path):
    good=payload([event()]);bad=payload([event(source_ids=['S99'])])
    model=RecordingModel([bad,good])
    result=run_extraction_arm({'arm':'semantic_plain','warmup':False},[{'id':'C1','story':'甲没有受伤。','gold_facts':'SECRET GOLD','review':'SECRET REVIEW'}],model,tmp_path)
    assert result['complete'] and len(model.seen)==2
    first,second=[batch[0][0] for batch in model.seen]
    assert first[0]==second[0] and second[0]['content']!=(ROOT/'agent_pipeline_v2/prompts/extractor_v1.txt').read_text(encoding='utf-8')
    assert first[1]==second[1] and len(second)==len(first)+1
    assert all(n==1536 for _,n in model.seen)
    assert 'SECRET' not in json.dumps(model.seen)


def test_semantic_existing_recovery_keeps_base_and_multiple_source_facts(tmp_path):
    first=payload([event(),event(event='甲仍然活着')])
    recovery=payload(ignored=[{'source_id':'S2','reason':'routine'}])
    model=RecordingModel([first,recovery])
    result=run_extraction_arm({'arm':'semantic_lookup5','warmup':False},[{'id':'C1','story':'甲没有受伤。甲整理衣领。'}],model,tmp_path)
    ex=result['rows'][0]['result']['benchmark_stages']['extraction']
    assert result['complete'] and model.enabled and len(model.seen)==2
    assert len(ex['events'])==2 and ex['ignored_spans']==[{'source_id':'S2','reason':'routine'}]
    assert model.seen[0][0][0][0]==model.seen[1][0][0][0]
    assert list(json.loads(model.seen[1][0][0][1]['content'])['target_spans'])==['S2']
    assert ex['calls'][0]['attempts'][1]['purpose']=='recover_missing_targets'


def test_semantic_does_not_relabel_conditional_or_belief_responses(tmp_path):
    rows=[event(event='甲相信乙没有受伤',modality='belief',mental_state='相信'),event(event='甲在结界关闭后才不会受伤',modality='conditional',conditions=['结界关闭后'],source_ids=['S2'])]
    model=RecordingModel([payload(rows)])
    result=run_extraction_arm({'arm':'semantic_plain','warmup':False},[{'id':'C1','story':'甲相信乙没有受伤，但甲在结界关闭后才不会受伤。'}],model,tmp_path)
    ex=result['rows'][0]['result']['benchmark_stages']['extraction']
    assert result['complete'] and len(model.seen)==1
    assert [e['modality'] for e in ex['events']]==['belief','conditional']
    assert ex['events'][1]['conditions']==['结界关闭后']


def test_d_cli_dispatch_no_model(monkeypatch,tmp_path):
    from agent_pipeline_v5.__main__ import main
    import agent_pipeline_v5.experiment as experiment
    seen=[];monkeypatch.setattr(experiment,'run_experiment',lambda args:seen.append(args) or 0)
    assert main(['run-d','--phase','screen','--order','candidate-first','--output',str(tmp_path)])==0
    assert seen[0].experiment=='5D0' and seen[0].order=='candidate-first'


def test_d_summary_uses_new_pending_and_refuses_screen_quality(tmp_path):
    from agent_pipeline_v5.experiment import summarize
    from agent_pipeline_v5.quality import reviewed_arm
    ensure_manifest(tmp_path,{'config':{'experiment':'5D0','phase':'screen','arms':['semantic_plain'],'case_ids':['C1']}})
    run_extraction_arm({'arm':'semantic_plain','warmup':False},[{'id':'C1','story':'甲没有受伤。'}],RecordingModel([payload([event()])]),tmp_path/'semantic_plain')
    summary=summarize(tmp_path)
    assert summary['arms']['semantic_plain']['pending']==1
    review=json.loads((tmp_path/'semantic_plain/review.json').read_text())
    assert review['schema_version']=='benchmark-5D0-review-v1' and review['cases']['C1']['E1']['label'] is None
    assert '语义契约' in (tmp_path/'summary.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='full'):reviewed_arm(tmp_path,'semantic_plain')


def test_offline_semantic_cli_does_not_import_model_packages():
    code="import sys;from agent_pipeline_v5.__main__ import main;\ntry:main(['run-d','--help'])\nexcept SystemExit as e:assert e.code==0\nassert not any(x in sys.modules for x in ('torch','transformers','sentence_transformers'))"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_d_prepare_strips_gold_and_freezes_prompt_and_reverse_order(tmp_path,monkeypatch):
    import agent_pipeline_v5.experiment as experiment
    from agent_pipeline_v5.identity import sha
    root=tmp_path/'source';root.mkdir();snapshot=tmp_path/'snapshot';snapshot.mkdir()
    def write(relative,value):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8');return p
    ids=list(experiment.SCREEN_IDS)+[f'C{i}' for i in range(20)]
    dataset=write('evaluation/story_benchmark_24_v2.json',json.dumps({'cases':[{'id':cid,'story':'甲没有受伤。','gold_facts':'SECRET'} for cid in ids]}))
    frozen={'evaluation/story_benchmark_24_v2.json':sha(dataset)}
    for rel in ('agent_pipeline_v2/extractor.py','agent_pipeline_v2/prompts/extractor_v1.txt','agent_pipeline_v2_1/extractor.py','agent_pipeline_v4/static.py','qwen_judge.py','agent_pipeline_v3/model.py','agent_pipeline_v2/model.py','agent_pipeline_v2/batch_llm.py'):
        frozen[rel]=sha(write(rel,'frozen'))
    write('agent_pipeline_v4_2/reports/benchmark_4_2B_20261017T033125Z_295aa1/manifest.json',json.dumps({'source':{'files':frozen}}))
    prompt=write('agent_pipeline_v5/prompts/extractor_semantic_v1.txt','test prompt')
    write('data/index/CURRENT','v1');write('data/index/versions/v1/config.json','{}')
    for name in ('evaluation/agent_pipeline_v2_1_lore_metadata.json','encoder.py','retrieval.py','index_store.py','uv.lock'):write(name,'{}')
    for name in ('config.json','generation_config.json','tokenizer_config.json'):(snapshot/name).write_text('{}')
    monkeypatch.setattr(experiment,'ROOT',root)
    monkeypatch.setattr(experiment,'snapshot_path',lambda p:snapshot)
    monkeypatch.setattr(experiment,'versions',lambda:{'transformers':'4.57.6'})
    cases,identity=experiment.prepare(root,root,'screen',experiment='5D0')
    assert cases==[{'id':cid,'story':'甲没有受伤。'} for cid in experiment.SCREEN_IDS]
    assert 'SECRET' not in json.dumps(cases)
    assert identity['files'][str(prompt.resolve())]==identity['config']['semantic_prompt_sha256']==sha(prompt)
    assert identity['config']['repair_policy']=='none'
    _,reverse=experiment.prepare(root,root,'screen','reverse','5D0')
    assert reverse['config']['arms']==list(reversed(identity['config']['arms']))
    _,candidate=experiment.prepare(root,root,'screen','candidate-first','5D0')
    assert candidate['config']['arms']==['semantic_plain','original_plain','semantic_lookup5','original_lookup5']
    for phase,kind in [('full','5D0'),('screen','5B'),('screen','5C1')]:
        with pytest.raises(ValueError):experiment.prepare(root,root,phase,'candidate-first',kind)
    ensure_manifest(tmp_path/'run',identity)
    prompt.write_text('changed prompt')
    _,changed=experiment.prepare(root,root,'screen',experiment='5D0')
    with pytest.raises(ValueError,match='changed'):ensure_manifest(tmp_path/'run',changed)
