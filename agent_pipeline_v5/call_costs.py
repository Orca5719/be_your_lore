"""Call-ID joins; missing telemetry stays unknown, never guessed by order."""
from collections import Counter, defaultdict
import json
from agent_pipeline_v4_1.audit import event_signature


def extraction(row):
    return row['result']['benchmark_stages']['extraction']


def _sum(records, name):
    values = [r.get(name) for r in records]
    return sum(values) if all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in values) else None


def summarize_calls(rows: list[dict], calls: list[dict], labels: dict) -> dict:
    linked = defaultdict(list)
    recovery = []
    raw_counts = Counter()
    for row in rows:
        cid = row['case_id']
        ex = extraction(row)
        by_signature = defaultdict(list)
        for ev in ex['events']:
            by_signature[event_signature(ev)].append(ev)
        for window in ex['calls']:
            for attempt in window['attempts']:
                call_id = attempt.get('timing', {}).get('call_id')
                purpose = attempt.get('purpose')
                category = ('coverage' if purpose == 'recover_missing_targets' else
                            'initial' if attempt.get('attempt') == 1 else
                            'retry' if isinstance(attempt.get('attempt'), int) else 'unknown')
                ref = {'case_id': cid, 'window_id': window['window_id'], 'category': category,
                       'attempt_status': attempt['status'], 'error': attempt.get('error')}
                if call_id:
                    linked[call_id].append(ref)
                try:
                    value = json.loads(attempt.get('raw_output', ''))
                except (ValueError, TypeError):
                    value = None
                if isinstance(value, dict):
                    for key in ('events', 'ignored_spans', 'non_event_span_ids'):
                        if isinstance(value.get(key), list):
                            raw_counts[key] += len(value[key])
                if category == 'coverage':
                    events = value.get('events') if isinstance(value, dict) else None
                    matches = []
                    unresolved = 0
                    for event in events if isinstance(events, list) else []:
                        found = by_signature.get(event_signature(event), []) if isinstance(event, dict) else []
                        if attempt['status'] == 'ok' and window['status'] == 'ok' and len(found) == 1:
                            eid = found[0]['id']
                            matches.append({'event_id': eid, **labels[cid][eid]})
                        else:
                            unresolved += 1
                    recovery.append({**ref, 'call_id': call_id,
                                     'raw_event_count': len(events) if isinstance(events, list) else None,
                                     'exact_final_matches': matches,
                                     'unmatched_raw_events': unresolved if isinstance(events, list) else None,
                                     'causal_gain_without_recovery': None})
    ids = [r.get('call_id') for r in calls]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('missing/duplicate inference call_id')
    groups = {k: [] for k in ('initial', 'retry', 'coverage', 'unknown')}
    details = []
    for call in calls:
        refs = linked.get(call['call_id'], [])
        # This audit's archived arms are batch=1. Never assign shared batch time to rows.
        valid = len(refs) == 1 and call.get('batch_size') == 1 and call.get('story_ids') == [refs[0]['case_id']]
        category = refs[0]['category'] if valid else 'unknown'
        groups[category].append(call)
        details.append({'call_id': call['call_id'], 'category': category, 'references': refs,
                        'input_tokens': call.get('input_tokens'), 'output_tokens': call.get('output_tokens'),
                        'total_seconds': call.get('total_time'), 'status': call.get('status')})
    totals = {}
    for category, records in groups.items():
        totals[category] = {'calls': len(records), **{
            out: _sum(records, source) for out, source in (
                ('total_seconds', 'total_time'), ('prefill_seconds', 'prefill_time'),
                ('decode_seconds', 'decode_time'), ('input_tokens', 'input_tokens'), ('output_tokens', 'output_tokens'))}}
    return {'groups': totals, 'calls': details, 'recovery': recovery,
            'unmatched_attempts': [ref['case_id'] + '/' + key for key, refs in linked.items() if key not in ids for ref in refs],
            'raw_output_object_counts': dict(raw_counts),
            'limitations': ['raw_output is after the frozen repair wrapper, not guaranteed original model bytes',
                           'object counts include failed/retried responses; they are not token proportions',
                           'exact final signature matches show provenance, not counterfactual causal gain',
                           'sum of call time is not story latency or end-to-end wall']}
