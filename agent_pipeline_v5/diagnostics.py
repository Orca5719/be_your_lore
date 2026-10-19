"""Read-only archive audit. Never imports inference runners or model packages."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil

from agent_pipeline_v4_1.audit import event_signature, validate_labels, score_extraction, score_end_to_end
from .identity import build_identity, sha
from .call_costs import summarize_calls, extraction
from .attribution import error_packets

A = Path('agent_pipeline_v4_2/reports/benchmark_4_2A_20261016T170530Z_ab70a2')
B = Path('agent_pipeline_v4_2/reports/benchmark_4_2B_20261017T033125Z_295aa1')
DATASET = Path('evaluation/story_benchmark_24_v2.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows_at(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def apply_corrections(review: dict, corrections: dict) -> dict:
    result = copy.deepcopy(review)
    seen = set()
    for entry in corrections['corrections']:
        key = (entry['case_id'], entry['event_id'])
        if key in seen:
            raise ValueError('duplicate review correction')
        seen.add(key)
        old = result['cases'][key[0]][key[1]]
        if (old['signature'], old['label'], old['gold_ids']) != (entry['signature'], entry['from_label'], entry['from_gold_ids']):
            raise ValueError('review correction does not match original signature/label')
        old.update(label=entry['to_label'], gold_ids=entry['to_gold_ids'],
                   note='Applied archived correction: ' + entry.get('reason', ''))
    return result


def validate_arm(cases: list[dict], rows: list[dict], downstream: list[dict], review: dict):
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate dataset case')
    if review.get('pending'):
        raise ValueError('review pending')
    if set(review['cases']) != {c['id'] for c in cases}:
        raise ValueError('extra/missing reviewed cases')
    validate_labels(cases, rows, review['cases'])
    expected = {c['id']: c for c in cases}
    if len(downstream) != len(cases) or {r['case_id'] for r in downstream} != set(expected):
        raise ValueError('missing/duplicate downstream cases')
    by_id = {r['case_id']: r for r in rows}
    for row in rows:
        ex = extraction(row)
        if row.get('status') != 'ok' or ex.get('status') != 'ok' or ex.get('processing_complete') is False:
            raise ValueError('incomplete extraction')
        if ex['text'] != expected[row['case_id']]['story']:
            raise ValueError('extraction story does not match dataset')
    for row in downstream:
        result = row['result']; judge = result['judge']; ret = judge['retrieval']
        if any(x.get('status') != 'ok' for x in (row, result, judge, ret)):
            raise ValueError('incomplete downstream')
        ex = extraction(by_id[row['case_id']])
        for events in (judge['events'], ret['extraction']['events']):
            if events != ex['events']:
                raise ValueError('downstream extraction changed')
        ids = [e['id'] for e in ex['events']]
        item_ids = [i['event_id'] for i in judge['items']]
        if len(item_ids) != len(ids) or set(item_ids) != set(ids):
            raise ValueError('missing/duplicate judge items')
        for item in judge['items']:
            if item.get('status') != 'ok':
                raise ValueError('failed judge item')
            if item.get('event') != next(e for e in ex['events'] if e['id'] == item['event_id']):
                raise ValueError('judge event differs from extraction')


def _history_check(root, manifest):
    checks = []
    if DATASET.as_posix() not in manifest['source']['files']:
        raise ValueError('frozen dataset hash missing from manifest')
    for rel, expected in manifest['source']['files'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            if rel == DATASET.as_posix():
                raise ValueError('frozen dataset resolves outside source root')
            checks.append({'path': rel, 'status': 'external_link_not_read', 'resolved_path': str(path)})
            continue
        status = 'missing' if not path.is_file() else 'match' if sha(path) == expected else 'changed'
        checks.append({'path': rel, 'status': status, 'expected_sha256': expected})
        if rel == DATASET.as_posix() and status != 'match':
            raise ValueError('dataset does not match frozen manifest')
    return checks


def audit_archives(source_root: Path, output_dir: Path) -> dict:
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError('source and output must not overlap')
    if output_dir.exists():
        raise ValueError('output already exists; use a new directory')
    paths = [source_root / DATASET, source_root / B / 'manifest.json',
             source_root / B / 'baseline_review_corrections.json', source_root / B / 'quality_reviewed.json']
    bundles = {}
    for name, base in [('baseline', A / 'baseline'), ('prompt_lookup', B / 'prompt_lookup')]:
        bundle = {key: source_root / base / file for key, file in (
            ('rows','extractor_runs.jsonl'), ('calls','inference_calls.jsonl'),
            ('review','review.json'), ('downstream','downstream_runs.jsonl'))}
        bundles[name] = bundle
        paths.extend(bundle.values())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    implementation = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).resolve().parents[1] / 'agent_pipeline_v4_1/audit.py']
    paths.extend(implementation)
    identity = build_identity({'schema_version':'benchmark-5A-v1','source_root':str(source_root)}, paths)
    manifest = read(source_root / B / 'manifest.json')
    historical = _history_check(source_root, manifest)
    # The historical manifest independently binds the baseline used in B.
    for key, field in [('rows','baseline_extractions'),('calls','baseline_calls'),('review','baseline_review'),('downstream','baseline_downstream')]:
        if sha(bundles['baseline'][key]) != manifest['baseline_files'][field]:
            raise ValueError('baseline archive no longer matches B manifest: ' + field)
    cases = read(source_root / DATASET)['cases']
    official = read(source_root / B / 'quality_reviewed.json')
    results, costs, packets = {}, {}, {}
    for name, bundle in bundles.items():
        rows, down, calls = (rows_at(bundle[k]) for k in ('rows','downstream','calls'))
        review = read(bundle['review'])
        if name == 'baseline':
            review = apply_corrections(review, read(source_root / B / 'baseline_review_corrections.json'))
        validate_arm(cases, rows, down, review)
        quality = {'extraction':score_extraction(cases, rows, review['cases']),
                   'end_to_end':score_end_to_end(cases, down, review['cases'])}
        if quality != official[name]:
            raise ValueError('recomputed quality differs from reviewed archive: ' + name)
        results[name] = quality
        costs[name] = summarize_calls(rows, calls, review['cases'])
        packets[name] = error_packets(cases, rows, down, review['cases'], quality['end_to_end'],
                                     {k:str(v) for k,v in bundle.items()})
    comparison = {}
    for metric, section in [('missed_gold','extraction'), ('missed_gold_contradictions','end_to_end')]:
        before = set(results['baseline'][section][metric]); after = set(results['prompt_lookup'][section][metric])
        comparison[metric] = {'new_in_prompt_lookup':sorted(after-before),'resolved_in_prompt_lookup':sorted(before-after),'shared':sorted(before&after)}
    warnings = ['Historical frozen score is reproduced, not independently certified gold truth.',
                'Dataset carries assistant-authored draft annotation status; suspected gold issues must be separately reviewed.',
                'No semantic replay of old generation; raw outputs have already passed the old repair wrapper.',
                'No GPU runs; current hashes freeze this audit, not a missing historical signature.',
                '4.2B and baseline timings are cross-run; only call costs are aggregated here.']
    if any(c['status'] != 'match' for c in historical):
        warnings.append('Some historical source files are missing/changed; see provenance.json.')
    # Rehash before writing: source files cannot change under a successful audit.
    if build_identity(identity['config'], paths) != identity:
        raise ValueError('source archive changed during audit')
    disk_parent = output_dir.parent
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    if shutil.disk_usage(disk_parent).free < 5 * 1024**3:
        raise ValueError('less than 5 GiB free on output volume')
    payload = {'quality': results, 'comparison': comparison, 'warnings': warnings}
    rendered = render_summary(payload, costs, packets)
    documents = {'manifest.json':identity,'audit.json':payload,'call_costs.json':costs,
                 'error_attribution.json':packets,'provenance.json':{'historical_source_checks':historical}}
    encoded = {name:json.dumps(value,ensure_ascii=False,indent=2) + '\n' for name,value in documents.items()}
    encoded['audit_summary.md'] = rendered
    encoded['decision.md'] = ('# 5A decision\n\n继续5B的最小短字段实验，但不把已由程序生成的ID算成新收益。\n'
        '5C必须以实际恢复产物为依据；当前实现已经只补缺失target，不应重复实现同一方案。\n'
        '准确率优先复核program_nonactual_scope与知识状态标记，并审查gold/证据争议；不据历史F1宣称4B到上限。\n'
        '以上是诊断优先级，不是效果证明；不修改历史标签或运行时行为。语义复核见单独的5A审阅记录。\n')
    if sum(len(s.encode('utf-8')) for s in encoded.values()) > 512 * 1024**2:
        raise ValueError('audit report exceeds 512 MiB budget')
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in encoded.items():
        temporary = output_dir / (name + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(output_dir / name)
    return payload


def render_summary(payload, costs, packets):
    lines = ['# Benchmark 5A 离线诊断', '', '仅复算历史报告；不运行模型，不修改历史标签。', '',
             '|组别|命中/总gold|TP/FP/FN|', '|---|---:|---:|']
    for name, q in payload['quality'].items():
        e,j=q['extraction'],q['end_to_end']
        lines.append(f"|{name}|{e['gold_hits']}/{e['gold_total']}|{j['tp']}/{j['fp']}/{j['fn']}|")
    lines += ['', '## 调用成本', '', '调用总时长不等于故事延迟或整组wall。retry为重试调用，不能声称仅JSON格式错误。', '',
              '|组别/用途|调用|总秒|输入tokens|输出tokens|', '|---|---:|---:|---:|---:|']
    for arm, cost in costs.items():
        for kind,g in cost['groups'].items():
            seconds = f"{g['total_seconds']:.3f}" if g['total_seconds'] is not None else 'unknown'
            lines.append(f"|{arm}/{kind}|{g['calls']}|{seconds}|{g['input_tokens']}|{g['output_tokens']}|")
    lines += ['', '## 当前候选的可观察错误边界', '', json.dumps(packets['prompt_lookup']['boundary_counts'], ensure_ascii=False),
              '', '完整原文、证据、Judge理由、gold引用保存在error_attribution.json；边界不等同最终根因。',
              '', '## 漏检集合变化', '', '```json', json.dumps(payload['comparison'],ensure_ascii=False,indent=2),'```', '', '## 限制', '']
    lines += ['- '+v for v in payload['warnings']]
    return '\n'.join(lines)+'\n'




