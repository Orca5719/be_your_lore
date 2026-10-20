"""Serial batch-one arms using the unchanged extractor, repair and queue."""
import hashlib
import json
from pathlib import Path
import time

from agent_pipeline_v2_2.runner import atomic_write_json, write_jsonl_atomic
from .extractor import WireBatchModel
from .resources import check_resources
from .identity import sha

ARMS = ('original_plain', 'compact_plain', 'original_lookup5', 'compact_lookup5')
C1_ARMS = ('original_plain','repaired_plain','original_lookup5','repaired_lookup5')
D0_ARMS = ('original_plain','semantic_plain','original_lookup5','semantic_lookup5')
RUN_ARMS = tuple(dict.fromkeys(ARMS+C1_ARMS+D0_ARMS))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if path.exists() else []


def ensure_manifest(directory: Path, identity: dict):
    path = directory / 'manifest.json'
    canonical = json.loads(json.dumps(identity, ensure_ascii=False))
    if path.exists():
        if read(path) != canonical:
            raise ValueError('configuration/code/data/runtime changed; use a new result directory')
    else:
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError('nonempty result directory without manifest')
        atomic_write_json(path, canonical)


def _checkpoint(output, rows, calls, wire):
    write_jsonl_atomic(output / 'inference_calls.jsonl', calls)
    write_jsonl_atomic(output / 'wire_calls.jsonl', wire)
    write_jsonl_atomic(output / 'extractor_runs.jsonl', rows)


def run_extraction_arm(config: dict, cases: list[dict], model, output_dir: Path) -> dict:
    from agent_pipeline_v2.extractor import extract_events
    from agent_pipeline_v2_1.extractor import ExtractionRepairLLM
    from agent_pipeline_v4.static import StaticBatcher, QueuedLLM
    config=dict(config)
    arm = config['arm']
    if arm not in RUN_ARMS:
        raise ValueError('invalid Benchmark 5 arm')
    output = Path(output_dir)
    if (output.parent/'manifest.json').exists():
        config['experiment_sha256']=sha(output.parent/'manifest.json')
    report_root = Path(config.get('report_root', output))
    check_resources(report_root, initial=True)
    ids = [c['id'] for c in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty/duplicate cases')
    arm_identity = {'config': config, 'cases': cases}
    manifest = output / 'manifest.json'
    if manifest.exists():
        ensure_manifest(output, arm_identity)
    else:
        ensure_manifest(output, arm_identity)
    rows, calls, wires = (jsonl(output / file) for file in ('extractor_runs.jsonl','inference_calls.jsonl','wire_calls.jsonl'))
    saved = [r['case_id'] for r in rows]
    if saved != ids[:len(saved)] or any(r.get('status') != 'ok' for r in rows):
        raise ValueError('partial/invalid prior arm is preserved; use a new directory')
    if rows:
        for name in ('inference_calls.jsonl','wire_calls.jsonl'):
            if not (output/name).is_file():
                raise ValueError('saved call log file missing: '+name)
        validate_logs(rows,calls,wires)
    completed = output / 'complete.json'
    if completed.exists():
        old = read(completed)
        for name, digest in old['files'].items():
            if sha(output / name) != digest:
                raise ValueError('completed arm files changed')
        if len(rows) != len(cases):
            raise ValueError('completed arm is incomplete')
        return {**old, 'rows': rows, 'calls': calls}
    resumed = (output / 'started.json').exists()
    atomic_write_json(output / 'started.json', {'started': True, 'resumed': resumed})
    model.enabled = arm.endswith('lookup5')
    wire_model = WireBatchModel(model, 'compact' if arm.startswith('compact') else 'original',
                               guard=lambda: check_resources(report_root),
                               repair_policy='schema_aliases_v1' if arm.startswith('repaired') else 'none',
                               prompt_policy='semantic_v1' if arm.startswith('semantic') else 'original')

    def one(case):
        with StaticBatcher(wire_model, 1) as batcher:
            result = extract_events(case['story'], device=model.device,
                llm=ExtractionRepairLLM(QueuedLLM(batcher, case['id'])))
        return result, batcher.records

    if config.get('warmup', True) and len(rows) < len(cases):
        warmup_start = time.perf_counter()
        warmup, _ = one({'id':'WARMUP', 'story':cases[0]['story']})
        if warmup['status'] != 'ok':
            atomic_write_json(output / 'failure.json', {'stage':'warmup','result':warmup})
            raise RuntimeError('warmup failed; see failure.json; no formal timing')
        wire_model.wire_records.clear()
        if hasattr(model, 'clear_traces'):
            model.clear_traces()
        atomic_write_json(output / 'warmup.json', {'seconds':time.perf_counter()-warmup_start,
                                                   'model_load_seconds':model.load_seconds})
    if hasattr(model, 'collector'):
        model.collector.resume_after(calls)
    started = time.perf_counter()
    for case in cases[len(rows):]:
        check_resources(report_root)
        story_start = time.perf_counter()
        try:
            result, records = one(case)
            row = {'case_id':case['id'], 'system':arm, 'status':result['status'],
                   'result':{'judge':{'events':result['events'],'items':[]},
                             'benchmark_stages':{'extraction':result}},
                   'stage_metrics':{'extraction':{'seconds':time.perf_counter()-story_start}}}
        except Exception as exc:
            row = {'case_id':case['id'],'system':arm,'status':'error','error':str(exc),
                   'result':{'benchmark_stages':{'extraction':{'status':'error','events':[],'calls':[]}}}}
            records = []
        rows.append(row)
        calls.extend(records)
        wires.extend(wire_model.wire_records)
        wire_model.wire_records.clear()
        _checkpoint(output, rows, calls, wires)
        print(f'[{arm}] {len(rows)}/{len(cases)} {case["id"]} {row["status"]}', flush=True)
        if hasattr(model, 'clear_traces'):
            model.clear_traces()
        if row['status'] != 'ok':
            break
    complete = len(rows) == len(cases) and all(r['status'] == 'ok' for r in rows)
    result = {'complete':complete, 'wall_seconds':time.perf_counter()-started if complete and not resumed else None,
              'resumed':resumed, 'files':{name:sha(output / name) for name in
                  ('extractor_runs.jsonl','inference_calls.jsonl','wire_calls.jsonl')}}
    atomic_write_json(output / ('complete.json' if complete else 'failure.json'), result)
    return {**result,'rows':rows,'calls':calls}


def validate_logs(rows, calls, wires):
    call_ids=[c.get('call_id') for c in calls]
    wire_ids=[w['call_id'] for w in wires if w.get('call_id') is not None]
    if len(call_ids)!=len(set(call_ids)) or len(wire_ids)!=len(set(wire_ids)):
        raise ValueError('duplicate saved call logs')
    for row in rows:
        for window in row['result']['benchmark_stages']['extraction']['calls']:
            for attempt in window['attempts']:
                call_id=attempt.get('timing',{}).get('call_id')
                if not call_id or call_id not in call_ids or call_id not in wire_ids:
                    raise ValueError('saved attempt missing inference/wire call log')
                record=calls[call_ids.index(call_id)]
                if record.get('story_ids')!=[row['case_id']]:
                    raise ValueError('saved call log belongs to another story')


def validate_arm_identity(directory, arm, rows):
    root=read(directory/'manifest.json')
    own=read(directory/arm/'manifest.json')
    if own['config'].get('arm')!=arm or own['config'].get('experiment_sha256')!=sha(directory/'manifest.json'):
        raise ValueError('arm identity differs from parent experiment')
    if [c['id'] for c in own['cases']]!=root['config']['case_ids']:
        raise ValueError('arm cases differ from experiment')
    stories={c['id']:c['story'] for c in own['cases']}
    for row in rows:
        if row.get('system')!=arm or (row.get('status')=='ok' and row['result']['benchmark_stages']['extraction'].get('text')!=stories.get(row['case_id'])):
            raise ValueError('row arm identity/story differs')
