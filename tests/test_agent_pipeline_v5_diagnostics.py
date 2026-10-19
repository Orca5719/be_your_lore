import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from agent_pipeline_v4_1.audit import event_signature, score_extraction, score_end_to_end
from agent_pipeline_v5.diagnostics import audit_archives, summarize_calls, apply_corrections, validate_arm
from agent_pipeline_v5.identity import build_identity


def event(eid='E1'):
    return dict(id=eid, actors=['甲'], event='甲知道秘密', mental_state=None, explicit=True,
                modality='belief', conditions=[], source_ids=['S1'], context_ids=[], check_reason='knowledge_relation')


def sample():
    ev=event()
    cases=[dict(id='C1',story='甲知道秘密。',gold_facts=[dict(id='G1',expected_verdict='矛盾',minimum_evidence_sets=[['K1']])])]
    attempt=dict(attempt=1,status='ok',raw_output=json.dumps({'events':[{k:v for k,v in ev.items() if k!='id'}]}),timing={'call_id':'L1'})
    extraction=dict(status='ok',text='甲知道秘密。',spans={'S1':{'text':'甲知道秘密。'}},events=[ev],calls=[dict(window_id=1,target_ids=['S1'],context_ids=[],status='ok',attempts=[attempt])])
    rows=[dict(case_id='C1',status='ok',result={'benchmark_stages':{'extraction':extraction}})]
    review=dict(pending=[],cases={'C1':{'E1':dict(signature=event_signature(ev),label='valid_checkable',gold_ids=['G1'])}})
    downstream=[dict(case_id='C1',status='ok',result={'status':'ok','judge':{'status':'ok','events':[ev],'items':[dict(event_id='E1',event=ev,status='ok',verdict='uncertain',origin='program_nonactual_scope',evidence=[{'id':'K1','text':'甲不知道秘密'}])], 'retrieval':{'status':'ok','extraction':extraction,'items':[]}}})]
    calls=[dict(call_id='L1',component='extractor',story_ids=['C1'],batch_size=1,input_tokens=10,output_tokens=5,total_time=1.,prefill_time=.2,decode_time=.8)]
    return cases,rows,downstream,review,calls


def test_identity_json_roundtrip(tmp_path):
    p=tmp_path/'x';p.write_text('x')
    identity=build_identity({'batches':(1,2)},[p])
    assert identity==json.loads(json.dumps(identity))
    p.write_text('y');assert build_identity({'batches':(1,2)},[p])!=identity


def test_review_corrections_guard_and_no_mutation():
    _,_,_,review,_=sample(); before=copy.deepcopy(review)
    entry=dict(case_id='C1',event_id='E1',signature=review['cases']['C1']['E1']['signature'],from_label='valid_checkable',from_gold_ids=['G1'],to_label='hallucinated',to_gold_ids=[])
    corrected=apply_corrections(review,{'corrections':[entry]})
    assert corrected['cases']['C1']['E1']['label']=='hallucinated' and review==before
    entry['signature']='stale'
    with pytest.raises(ValueError):apply_corrections(review,{'corrections':[entry]})


@pytest.mark.parametrize('damage',['duplicate','missing','stale','pending','partial','downstream_changed'])
def test_validation_rejects_invalid_archive(damage):
    cases,rows,down,review,_=sample()
    if damage=='duplicate': rows.append(copy.deepcopy(rows[0]))
    if damage=='missing': rows=[]
    if damage=='stale': review['cases']['C1']['E1']['signature']='bad'
    if damage=='pending':review['pending']=['C1/E1']
    if damage=='partial':down[0]['status']='partial'
    if damage=='downstream_changed':down[0]['result']['judge']['events']=[event('E2')]
    with pytest.raises(ValueError):validate_arm(cases,rows,down,review)


def test_costs_join_by_call_id_and_unknown_metrics():
    _,rows,_,review,calls=sample()
    calls[0].pop('prefill_time')
    out=summarize_calls(rows,calls,review['cases'])
    assert out['groups']['initial']['calls']==1
    assert out['groups']['initial']['prefill_seconds'] is None
    assert out['groups']['initial']['total_seconds']==1
    assert out['groups']['initial']['output_tokens']==5
    calls[0]['call_id']='unlinked'
    out=summarize_calls(rows,calls,review['cases'])
    assert out['groups']['unknown']['calls']==1
    assert out['unmatched_attempts']==['C1/L1']


def test_recovery_event_provenance_not_span_guess():
    _,rows,_,review,calls=sample()
    a=rows[0]['result']['benchmark_stages']['extraction']['calls'][0]['attempts'][0]
    a['purpose']='recover_missing_targets'
    out=summarize_calls(rows,calls,review['cases'])
    assert out['groups']['coverage']['calls']==1
    assert out['recovery'][0]['exact_final_matches'][0]['label']=='valid_checkable'
    raw=json.loads(a['raw_output']);raw['events'][0]['event']='不同的事件';a['raw_output']=json.dumps(raw)
    out=summarize_calls(rows,calls,review['cases'])
    assert out['recovery'][0]['unmatched_raw_events']==1
    assert out['recovery'][0]['exact_final_matches']==[]


def test_missing_archive_and_overlapping_output(tmp_path):
    src=tmp_path/'src';src.mkdir()
    with pytest.raises(ValueError):audit_archives(src,src/'output')
    with pytest.raises((ValueError,FileNotFoundError)):audit_archives(src,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_cli_import_does_not_load_model():
    code="import sys; import agent_pipeline_v5.diagnostics; assert not any(x in sys.modules for x in ['torch','transformers','numpy','sentence_transformers'])"
    subprocess.run([sys.executable,'-c',code],check=True)


def make_archive(root):
    from agent_pipeline_v5.diagnostics import A,B,DATASET
    from agent_pipeline_v5.identity import sha
    cases,rows,down,review,calls=sample()
    def write(p,value,jsonl=False):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join(json.dumps(x) for x in value) if jsonl else json.dumps(value),encoding='utf-8')
    write(root/DATASET,{'cases':cases})
    for base in [A/'baseline',B/'prompt_lookup']:
        for name,value in [('extractor_runs.jsonl',rows),('downstream_runs.jsonl',down),('inference_calls.jsonl',calls)]:write(root/base/name,value,True)
        write(root/base/'review.json',review)
    write(root/B/'baseline_review_corrections.json',{'corrections':[]})
    quality={'extraction':score_extraction(cases,rows,review['cases']),'end_to_end':score_end_to_end(cases,down,review['cases'])}
    write(root/B/'quality_reviewed.json',{'baseline':quality,'prompt_lookup':quality})
    baseline_files={k:sha(root/A/'baseline'/v) for k,v in [('baseline_extractions','extractor_runs.jsonl'),('baseline_calls','inference_calls.jsonl'),('baseline_review','review.json'),('baseline_downstream','downstream_runs.jsonl')]}
    write(root/B/'manifest.json',{'source':{'files':{DATASET.as_posix():sha(root/DATASET)}},'baseline_files':baseline_files})


def test_end_to_end_audit_new_parent_and_preserves_source(tmp_path):
    src=tmp_path/'source';make_archive(src)
    before=build_identity({},list(src.rglob('*.*')))
    output=tmp_path/'not-yet-created'/'audit'
    result=audit_archives(src,output)
    assert result['quality']['prompt_lookup']['end_to_end']['fn']==1
    assert build_identity({},list(src.rglob('*.*')))==before
    assert (output/'error_attribution.json').is_file()
    with pytest.raises(ValueError):audit_archives(src,output)


def test_duplicate_call_id_rejected():
    _,rows,_,review,calls=sample()
    with pytest.raises(ValueError):summarize_calls(rows,calls+calls,review['cases'])


def test_dataset_binding_is_required(tmp_path):
    from agent_pipeline_v5.diagnostics import B
    src=tmp_path/'src';make_archive(src)
    p=src/B/'manifest.json';value=json.loads(p.read_text());value['source']['files']={};p.write_text(json.dumps(value))
    with pytest.raises(ValueError):audit_archives(src,tmp_path/'out')


def test_corrupt_quality_rejected(tmp_path):
    from agent_pipeline_v5.diagnostics import B
    src=tmp_path/'src';make_archive(src)
    p=src/B/'quality_reviewed.json';value=json.loads(p.read_text());value['prompt_lookup']['end_to_end']['tp']=999;p.write_text(json.dumps(value))
    with pytest.raises(ValueError):audit_archives(src,tmp_path/'out')


def test_scope_and_missing_boundaries_are_not_model_failures():
    from agent_pipeline_v5.attribution import error_packets
    cases,rows,down,review,_=sample()
    quality=score_end_to_end(cases,down,review['cases'])
    p=error_packets(cases,rows,down,review['cases'],quality,{})
    assert p['items'][0]['observed_failure_boundary']=='program_scope_gate'
    assert p['items'][0]['first_confirmed_error'] is None
    assert p['items'][0]['evidence_checks'][0]['annotated_minimum_present'] is True


def test_low_disk_refuses_before_output(tmp_path,monkeypatch):
    import agent_pipeline_v5.diagnostics as diagnostics
    from types import SimpleNamespace
    src=tmp_path/'src';make_archive(src)
    monkeypatch.setattr(diagnostics.shutil,'disk_usage',lambda p:SimpleNamespace(free=100))
    with pytest.raises(ValueError,match='GiB'):audit_archives(src,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_linked_dataset_cannot_skip_frozen_hash(tmp_path, monkeypatch):
    from agent_pipeline_v5.diagnostics import _history_check, DATASET
    root=tmp_path/'src';root.mkdir()
    external=tmp_path/'changed.json';external.write_text('{}')
    original=Path.resolve
    def redirected(path,*args,**kwargs):
        return external if path==root/DATASET else original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',redirected)
    with pytest.raises(ValueError,match='dataset'):
        _history_check(root,{'source':{'files':{DATASET.as_posix():'old-hash'}}})
