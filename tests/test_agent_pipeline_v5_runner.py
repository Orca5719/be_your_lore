import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from agent_pipeline_v5.resources import check_resources, ResourceLimit
from agent_pipeline_v5.runner import run_extraction_arm, ensure_manifest, ARMS
from agent_pipeline_v5.extractor import WireBatchModel


def test_resource_gate_checks_temp_output_and_report_budget(tmp_path,monkeypatch):
    import agent_pipeline_v5.resources as resources
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda p:SimpleNamespace(free=10*1024**3))
    assert check_resources(tmp_path,tmp_path,initial=True)['report_bytes']==0
    (tmp_path/'large').write_bytes(b'x'*101)
    with pytest.raises(ResourceLimit):check_resources(tmp_path,tmp_path,report_budget=100)
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda p:SimpleNamespace(free=1024))
    with pytest.raises(ResourceLimit):check_resources(tmp_path,tmp_path)


def test_manifest_json_roundtrip_and_config_drift(tmp_path):
    ensure_manifest(tmp_path,{'sizes':(1,2)})
    ensure_manifest(tmp_path,{'sizes':[1,2]})
    with pytest.raises(ValueError):ensure_manifest(tmp_path,{'sizes':[4]})


class FakeModel:
    device='cpu'
    load_seconds=0
    enabled=False
    def __init__(self):self.last_batch_generation={};self.n=0
    def _generate_batch(self,batches,max_new_tokens=1536):
        self.n+=1
        from agent_pipeline_v5.wire import REQUIRED
        payload=json.loads(batches[0][1]['content'])
        target=next(iter(payload['target_spans']))
        event=dict(actors=['雷'],event='雷没有受伤',source_ids=[target],modality='observed',check_reason='mechanism')
        if '每个event必填' not in batches[0][0]['content']:
            event.update(explicit=True,mental_state=None,conditions=[],context_ids=[])
        raw=json.dumps({'events':[event],'ignored_spans':[],'non_event_span_ids':[]},ensure_ascii=False)
        profile=dict(call_id=f'L{self.n}',status='ok',input_tokens=12,output_tokens=6,total_time=.1,prefill_time=.01,decode_time=.09,batch_size=1)
        self.last_batch_generation=dict(call_id=f'L{self.n}',profile=profile,input_tokens=[12],generated_tokens=[6],seconds=.1,truncated_indices=[])
        return [raw]
    def clear_traces(self):pass


def test_four_arms_frozen_extractor_integration_and_resume(tmp_path):
    case={'id':'C1','story':'雷没有受伤。'}
    outputs=[]
    for arm in ARMS:
        fake=FakeModel(); config={'arm':arm,'warmup':False}
        result=run_extraction_arm(config,[case],fake,tmp_path/arm)
        assert result['complete'] and len(result['rows'])==1
        outputs.append(result['rows'][0]['result']['benchmark_stages']['extraction']['events'])
        assert fake.enabled==arm.endswith('lookup5')
        assert (tmp_path/arm/'wire_calls.jsonl').is_file()
        again=run_extraction_arm(config,[case],fake,tmp_path/arm)
        assert again['wall_seconds']==result['wall_seconds'] and fake.n==1
    assert all(events==outputs[0] for events in outputs)


def test_stop_after_first_failed_story_preserves_checkpoint(tmp_path):
    fake=FakeModel()
    def fail(*a,**k):raise RuntimeError('CUDA out of memory')
    fake._generate_batch=fail
    result=run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'},{'id':'C2','story':'雷没有受伤。'}],fake,tmp_path/'arm')
    assert not result['complete'] and len(result['rows'])==1
    with pytest.raises(ValueError):run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],fake,tmp_path/'arm')



def test_screen_summary_never_publishes_quality(tmp_path):
    from agent_pipeline_v5.experiment import summarize
    ensure_manifest(tmp_path,{'config':{'arms':[ARMS[0]],'phase':'screen','case_ids':['C1']}})
    run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],FakeModel(),tmp_path/ARMS[0])
    value=summarize(tmp_path)
    assert value['phase']=='screen' and 'quality' not in value
    assert value['arms'][ARMS[0]]['pending']==1


def test_original_inference_case_never_receives_gold(tmp_path,monkeypatch):
    import agent_pipeline_v5.experiment as experiment
    # A full prepare fixture would require model/index assets: the inference boundary
    # is separately exercised with an event asserting all supplied user text keys.
    fake=FakeModel()
    old=fake._generate_batch
    def checked(batches,**kwargs):
        payload=json.loads(batches[0][1]['content'])
        assert set(payload)=={'target_spans','context_spans'}
        assert 'gold_facts' not in json.dumps(payload)
        return old(batches,**kwargs)
    fake._generate_batch=checked
    out=run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。','gold_facts':['SECRET']}],fake,tmp_path/'arm')
    assert out['complete']


def test_formal_scoring_rejects_screen(tmp_path):
    from agent_pipeline_v5.quality import reviewed_arm
    ensure_manifest(tmp_path,{'config':{'phase':'screen'}})
    with pytest.raises(ValueError,match='full'):reviewed_arm(tmp_path,ARMS[0])


def test_interrupted_resume_wall_is_unknown(tmp_path):
    out=tmp_path/'arm';out.mkdir()
    (out/'started.json').write_text('{}')
    (out/'manifest.json').write_text(json.dumps({'config':{'arm':ARMS[0],'warmup':False},'cases':[{'id':'C1','story':'雷没有受伤。'}]}))
    value=run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],FakeModel(),out)
    assert value['complete'] and value['wall_seconds'] is None


def full_fixture(tmp_path):
    from agent_pipeline_v5.experiment import fresh_review
    from agent_pipeline_v5.identity import sha
    from agent_pipeline_v4_1.audit import event_signature
    src=tmp_path/'source';(src/'evaluation').mkdir(parents=True)
    cases=[{'id':f'C{i:02}','story':'雷没有受伤。','gold_facts':[{'id':'G1','expected_verdict':'一致'}]} for i in range(24)]
    dataset=src/'evaluation/story_benchmark_24_v2.json';dataset.write_text(json.dumps({'cases':cases}),encoding='utf-8')
    out=tmp_path/'output'
    ensure_manifest(out,{'config':{'phase':'full','source_root':str(src),'arms':[ARMS[0]],'case_ids':[c['id'] for c in cases]},'files':{str(dataset.resolve()):sha(dataset)}})
    result=run_extraction_arm({'arm':ARMS[0],'warmup':False},cases,FakeModel(),out/ARMS[0])
    fresh_review(out,ARMS[0],result['rows'])
    path=out/ARMS[0]/'review.json';review=json.loads(path.read_text())
    for labels in review['cases'].values():
        for label in labels.values():label.update(label='valid_checkable',gold_ids=['G1'])
    review['pending']=[];path.write_text(json.dumps(review),encoding='utf-8')
    return out,result,review


def test_full_quality_gate_reviews_and_partial_results(tmp_path):
    from agent_pipeline_v5.quality import reviewed_arm,score
    out,result,review=full_fixture(tmp_path)
    assert len(reviewed_arm(out,ARMS[0])[0])==24
    value=score(out)
    assert value[ARMS[0]]['extraction']['recall']==1
    assert value[ARMS[0]]['end_to_end'] is None
    path=out/ARMS[0]/'review.json';review['pending']=['C00/E1'];path.write_text(json.dumps(review))
    with pytest.raises(ValueError):reviewed_arm(out,ARMS[0])


def test_completed_arm_missing_rows_cannot_be_skipped(tmp_path):
    from agent_pipeline_v5.experiment import summarize
    out,result,review=full_fixture(tmp_path)
    (out/ARMS[0]/'extractor_runs.jsonl').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):summarize(out)


def test_downstream_bound_to_run_manifest(tmp_path):
    from agent_pipeline_v5.quality import score
    from agent_pipeline_v5.identity import sha
    out,result,review=full_fixture(tmp_path)
    arm=out/ARMS[0]
    (arm/'downstream_manifest.json').write_text(json.dumps({'review_sha256':sha(arm/'review.json'),'extraction_sha256':sha(arm/'extractor_runs.jsonl'),'run_manifest_sha256':'stale'}))
    (arm/'downstream_runs.jsonl').write_text('')
    with pytest.raises(ValueError,match='manifest'):score(out)


def test_offline_help_import_does_not_load_torch():
    import subprocess,sys
    code="import sys;from agent_pipeline_v5.__main__ import main; main(['validate-b','--help'])"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0
    code="import sys; import agent_pipeline_v5.experiment,agent_pipeline_v5.quality; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('damage',['wire_missing','calls_missing','wire_mismatch'])
def test_resume_requires_complete_saved_call_logs(tmp_path,damage):
    out,result,review=full_fixture(tmp_path)
    arm=out/ARMS[0];(arm/'complete.json').unlink()
    if damage=='wire_missing':(arm/'wire_calls.jsonl').unlink()
    elif damage=='calls_missing':(arm/'inference_calls.jsonl').unlink()
    else:
        records=[json.loads(x) for x in (arm/'wire_calls.jsonl').read_text().splitlines()]
        records[0]['call_id']='orphan';(arm/'wire_calls.jsonl').write_text('\n'.join(json.dumps(x) for x in records))
    identity=json.loads((arm/'manifest.json').read_text())
    with pytest.raises(ValueError,match='log|call'):
        run_extraction_arm(identity['config'],identity['cases'],FakeModel(),arm)


def test_copied_arm_cannot_be_scored_under_another_name(tmp_path):
    import shutil
    from agent_pipeline_v5.identity import sha
    from agent_pipeline_v5.quality import reviewed_arm
    out,result,review=full_fixture(tmp_path)
    root=json.loads((out/'manifest.json').read_text());root['config']['arms'].append(ARMS[1]);(out/'manifest.json').write_text(json.dumps(root))
    shutil.copytree(out/ARMS[0],out/ARMS[1])
    with pytest.raises(ValueError,match='arm|identity'):
        reviewed_arm(out,ARMS[1])


def test_shared_model_root_restored_even_if_constructor_fails(tmp_path,monkeypatch):
    import sys
    import qwen_judge
    import agent_pipeline_v5.model as shared
    old=qwen_judge.ROOT
    monkeypatch.setattr(shared,'snapshot_path',lambda p:tmp_path)
    monkeypatch.setattr(shared,'versions',lambda:{'transformers':'4.57.6'})
    def fake(*args,**kwargs):
        assert qwen_judge.ROOT==tmp_path.resolve()
        raise RuntimeError('constructor failure')
    monkeypatch.setitem(sys.modules,'agent_pipeline_v4_2.speculative_model',SimpleNamespace(PromptLookupExtractorModel=fake))
    with pytest.raises(RuntimeError,match='constructor'):shared.load_model(tmp_path)
    assert qwen_judge.ROOT==old


def test_failed_arm_still_produces_status_summary(tmp_path):
    from agent_pipeline_v5.experiment import summarize
    ensure_manifest(tmp_path,{'config':{'arms':[ARMS[0]],'phase':'screen','case_ids':['C1']}})
    fake=FakeModel()
    def fail(*args,**kwargs):raise RuntimeError('CUDA out of memory')
    fake._generate_batch=fail
    result=run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],fake,tmp_path/ARMS[0])
    assert not result['complete']
    summary=summarize(tmp_path)
    assert not summary['arms'][ARMS[0]]['complete']
    assert summary['arms'][ARMS[0]]['status_counts']['error']==1


def test_unmeasured_summary_metrics_remain_unknown(tmp_path):
    from agent_pipeline_v5.experiment import summarize
    ensure_manifest(tmp_path,{'config':{'arms':[ARMS[0]],'phase':'screen','case_ids':['C1']}})
    run_extraction_arm({'arm':ARMS[0],'warmup':False},[{'id':'C1','story':'雷没有受伤。'}],FakeModel(),tmp_path/ARMS[0])
    arm=summarize(tmp_path)['arms'][ARMS[0]]
    assert arm['ttft_p50_ms'] is None and arm['peak_allocated_bytes'] is None
    assert arm['story_p50_seconds']>=0 and arm['coverage_requests']==0
