import copy
import json
from pathlib import Path
import pytest

from agent_pipeline_v5.repair import repair_payload, repair_raw
from agent_pipeline_v5.runner import run_extraction_arm, C1_ARMS
from agent_pipeline_v5.extractor import WireBatchModel


def event(**changes):
    value=dict(actors=['雷'],event='雷没有受伤',source_ids=['S1'],context_ids=[],conditions=[],
               mental_state=None,explicit=True,modality='observed',check_reason='mechanism')
    value.update(changes);return value


def envelope(events=None,ignored=None):
    return dict(events=events or [],ignored_spans=ignored or [],non_event_span_ids=[])


def test_allowlisted_aliases_preserve_every_value_and_input():
    row=event();row['mentalological_state']=row.pop('mental_state')
    original=envelope([row],[{'source_Ids':'S2','reason':'routine'}]);snapshot=copy.deepcopy(original)
    repaired,actions=repair_payload(original)
    assert original==snapshot
    assert repaired['events'][0]==event()
    assert repaired['ignored_spans']==[{'source_id':'S2','reason':'routine'}]
    assert len(actions)==2


def test_collision_never_overwrites_or_discards_fields():
    value=envelope([event(mentalological_state='恐惧')],[{'source_id':'S1','source_Ids':'S2','reason':'routine'}])
    result,actions=repair_payload(value)
    assert result==value and actions==[]


@pytest.mark.parametrize('raw',['{broken','{"events": [','null','[]'])
def test_truncated_or_invalid_reply_not_salvaged(raw):
    assert repair_raw(raw)==(raw,[])


def test_unknown_fields_semantics_and_same_source_facts_unchanged():
    value=envelope([event(check_reason='process_detail',actors=['未知主体']),event(event='雷可能死亡',modality='conditional',conditions=['结界破裂'])])
    value['events'][0]['mentalological_state']=value['events'][0].pop('mental_state')
    result,actions=repair_payload(value)
    assert result['events'][0]['check_reason']=='process_detail'
    assert result['events'][0]['actors']==['未知主体']
    assert result['events'][1]==value['events'][1] and len(result['events'])==2
    assert len(actions)==1


def test_unknown_alias_not_guessed():
    value=envelope([event(extra_field='x')],[{'sourceid':'S1','reason':'routine'}])
    assert repair_payload(value)==(value,[])


class ScriptedModel:
    device='cpu';load_seconds=0;enabled=False
    def __init__(self,replies):self.replies=replies;self.n=0;self.last_batch_generation={}
    def _generate_batch(self,batches,max_new_tokens=1536):
        raw=self.replies[min(self.n,len(self.replies)-1)];self.n+=1
        profile=dict(call_id=f'L{self.n}',status='ok',input_tokens=20,output_tokens=10,total_time=.1,prefill_time=.01,decode_time=.09,batch_size=1)
        self.last_batch_generation=dict(call_id=f'L{self.n}',profile=profile,input_tokens=[20],generated_tokens=[10],seconds=.1,truncated_indices=[])
        return [raw]
    def clear_traces(self):pass


def test_c1_four_arms_repair_without_extra_model_call(tmp_path):
    broken=json.dumps(envelope(ignored=[{'source_Ids':'S1','reason':'routine'}]))
    valid=json.dumps(envelope(ignored=[{'source_id':'S1','reason':'routine'}]))
    for arm in C1_ARMS:
        model=ScriptedModel([broken,valid]);out=run_extraction_arm({'arm':arm,'warmup':False},[{'id':'C1','story':'雷走了。'}],model,tmp_path/arm)
        assert out['complete']
        assert model.n==(1 if arm.startswith('repaired') else 2)
        traces=[json.loads(x) for x in (tmp_path/arm/'wire_calls.jsonl').read_text(encoding='utf-8').splitlines()]
        assert traces[0]['raw_output']==broken
        assert bool(traces[0].get('schema_repairs')) is arm.startswith('repaired')
        assert model.enabled is arm.endswith('lookup5')


def test_semantic_error_uses_original_retry_budget(tmp_path):
    bad=json.dumps(envelope([event(check_reason='process_detail')]))
    good=json.dumps(envelope([event()]))
    outputs=[]
    for arm in ('original_plain','repaired_plain'):
        model=ScriptedModel([bad,good])
        out=run_extraction_arm({'arm':arm,'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],model,tmp_path/arm)
        assert out['complete'] and model.n==2
        outputs.append(out['rows'][0]['result']['benchmark_stages']['extraction']['events'])
    assert outputs[0]==outputs[1]


def test_existing_coverage_retains_multiple_events(tmp_path):
    first=json.dumps(envelope([event(),event(event='雷仍然活着')]))
    recovery=json.dumps(envelope(ignored=[{'source_id':'S2','reason':'routine'}]))
    results=[]
    for arm in ('original_plain','repaired_plain'):
        model=ScriptedModel([first,recovery]);out=run_extraction_arm({'arm':arm,'warmup':False},[{'id':'C1','story':'雷没有受伤。雷走了。'}],model,tmp_path/arm)
        assert out['complete'] and model.n==2
        ex=out['rows'][0]['result']['benchmark_stages']['extraction']
        assert len(ex['events'])==2
        assert ex['calls'][0]['attempts'][1]['purpose']=='recover_missing_targets'
        results.append(ex['events'])
    assert results[0]==results[1]


def replay_fixture(tmp_path,recovery=False):
    from agent_pipeline_v5.runner import ensure_manifest
    from agent_pipeline_v5.identity import sha
    import agent_pipeline_v2.extractor as frozen
    root=tmp_path/'history'
    ensure_manifest(root,{'config':{'arms':['original_plain'],'phase':'screen','case_ids':['C1']},'files':{str(Path(frozen.__file__).resolve()):sha(Path(frozen.__file__))}})
    target='S2' if recovery else 'S1'
    bad=json.dumps(envelope(ignored=[{'source_Ids':target,'reason':'routine'}]))
    good=json.dumps(envelope([event()] if recovery else [],[{'source_id':target,'reason':'routine'}]))
    replies=([json.dumps(envelope([event()]))] if recovery else [])+[bad,good]
    run_extraction_arm({'arm':'original_plain','warmup':False},[{'id':'C1','story':'雷没有受伤。雷走了。' if recovery else '雷走了。'}],ScriptedModel(replies),root/'original_plain')
    return root


def test_replay_is_readonly_and_reports_schema_not_quality(tmp_path):
    from agent_pipeline_v5.repair import replay_archive
    from agent_pipeline_v5.identity import sha
    root=replay_fixture(tmp_path)
    before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    value=replay_archive(root,tmp_path/'replay')
    assert value['arms']['original_plain']['changed_responses']==1
    assert value['arms']['original_plain']['schema_valid_after']==1
    assert 'quality' not in value
    assert {str(p):sha(p) for p in root.rglob('*') if p.is_file()}==before
    with pytest.raises(ValueError):replay_archive(root,root/'replay')


@pytest.mark.parametrize('damage',['truncated_rows','missing_binding','tampered_wire'])
def test_replay_rejects_incomplete_or_tampered_archive(tmp_path,damage):
    from agent_pipeline_v5.repair import replay_archive
    from agent_pipeline_v5.identity import sha
    root=replay_fixture(tmp_path);arm=root/'original_plain'
    complete=json.loads((arm/'complete.json').read_text())
    if damage=='truncated_rows':
        (arm/'extractor_runs.jsonl').write_text('')
        complete['files']['extractor_runs.jsonl']=sha(arm/'extractor_runs.jsonl')
    elif damage=='missing_binding':del complete['files']['wire_calls.jsonl']
    else:(arm/'wire_calls.jsonl').write_text('tampered')
    (arm/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError):replay_archive(root,tmp_path/'replay')
    assert not (tmp_path/'replay').exists()


def test_c1_summary_counts_actions_and_requires_new_reviews(tmp_path):
    from agent_pipeline_v5.runner import ensure_manifest
    from agent_pipeline_v5.experiment import summarize
    from agent_pipeline_v5.quality import reviewed_arm
    ensure_manifest(tmp_path,{'config':{'experiment':'5C1','phase':'screen','arms':['repaired_plain'],'case_ids':['C1']}})
    row=event();row['mentalological_state']=row.pop('mental_state')
    run_extraction_arm({'arm':'repaired_plain','warmup':False},[{'id':'C1','story':'雷没有受伤。'}],ScriptedModel([json.dumps(envelope([row]))]),tmp_path/'repaired_plain')
    result=summarize(tmp_path)['arms']['repaired_plain']
    assert result['schema_repair_calls']==1 and result['schema_repair_actions']==1
    review=json.loads((tmp_path/'repaired_plain/review.json').read_text())
    assert review['schema_version']=='benchmark-5C1-review-v1' and len(review['pending'])==1
    assert '# Benchmark 5C1' in (tmp_path/'summary.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='full'):reviewed_arm(tmp_path,'repaired_plain')


def test_run_c_dispatches_c1_without_loading_model(monkeypatch,tmp_path):
    from agent_pipeline_v5.__main__ import main
    import agent_pipeline_v5.experiment as experiment
    seen=[]
    monkeypatch.setattr(experiment,'run_experiment',lambda args:seen.append(args) or 0)
    assert main(['run-c','--phase','screen','--output',str(tmp_path)])==0
    assert seen[0].experiment=='5C1' and seen[0].output==tmp_path


def test_c1_offline_imports_do_not_load_model():
    import subprocess,sys
    code="import sys;import agent_pipeline_v5.__main__,agent_pipeline_v5.repair_audit;assert not any(x in sys.modules for x in ['torch','transformers','sentence_transformers'])"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_replay_uses_missing_targets_for_recovery_attempt(tmp_path):
    from agent_pipeline_v5.repair import replay_archive
    root=replay_fixture(tmp_path,recovery=True)
    stats=replay_archive(root,tmp_path/'replay')['arms']['original_plain']
    assert stats['changed_responses']==1 and stats['schema_valid_after']==1


def test_replay_rejects_unbound_validator(tmp_path):
    from agent_pipeline_v5.repair import replay_archive
    root=replay_fixture(tmp_path)
    manifest=json.loads((root/'manifest.json').read_text());manifest['files']={}
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='validator'):replay_archive(root,tmp_path/'replay')
