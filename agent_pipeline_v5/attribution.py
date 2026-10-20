"""Evidence packets for every scorer-defined false positive / missed gold."""
from collections import Counter


def error_packets(cases, rows, downstream, labels, quality, source_paths):
    cases = {c['id']: c for c in cases}
    extractions = {r['case_id']: r['result']['benchmark_stages']['extraction'] for r in rows}
    judges = {r['case_id']: r['result']['judge'] for r in downstream}
    packets = []
    for kind, keys in [('FP', quality['false_positive_events']), ('FN', quality['missed_gold_contradictions'])]:
        for key in keys:
            cid, ident = key.split('/')
            ex, judge = extractions[cid], judges[cid]
            if kind == 'FP':
                eids = [ident]
                gids = labels[cid][ident]['gold_ids']
            else:
                gids = [ident]
                eids = [eid for eid, v in labels[cid].items() if v['label'] == 'valid_checkable' and ident in v['gold_ids']]
            events = [e for e in ex['events'] if e['id'] in eids]
            golds = [g for g in cases[cid]['gold_facts'] if g['id'] in gids]
            items = [i for i in judge['items'] if i['event_id'] in eids]
            retrieval = [i for i in judge['retrieval']['items'] if i['event_id'] in eids]
            checks = []
            for g in golds:
                for item in items:
                    evidence = {e['id'] for e in item.get('evidence', [])}
                    sets = g.get('minimum_evidence_sets', [])
                    checks.append({'gold_id': g['id'], 'event_id': item['event_id'],
                                   'annotated_minimum_present': any(set(s) <= evidence for s in sets) if sets else None})
            if kind == 'FN' and not eids:
                boundary = 'extraction_missing'
            elif kind == 'FN' and items and all(i.get('origin') == 'program_nonactual_scope' for i in items):
                boundary = 'program_scope_gate'
            elif kind == 'FP' and labels[cid][ident]['label'] != 'valid_checkable':
                boundary = 'extraction_nonvalid_then_judge'
            else:
                boundary = 'requires_semantic_review'
            packets.append({'key': kind + ':' + key, 'kind': kind, 'case_id': cid,
                            'story': cases[cid]['story'], 'annotation_status': cases[cid].get('annotation_status'),
                            'gold_facts': golds, 'events': events,
                            'event_reviews': {e: labels[cid][e] for e in eids},
                            'source_spans': ex.get('spans', {}), 'retrieval': retrieval, 'judge_items': items,
                            'evidence_checks': checks, 'observed_failure_boundary': boundary,
                            'first_confirmed_error': 'extraction' if boundary == 'extraction_missing' else None,
                            'root_cause_status': 'needs_semantic_review', 'follow_on_errors': [],
                            'evidence_paths': source_paths})
    return {'items': packets, 'counts': dict(Counter(p['kind'] for p in packets)),
            'boundary_counts': dict(Counter(p['observed_failure_boundary'] for p in packets)),
            'limitations': ['annotated evidence-ID recall is not semantic evidence sufficiency',
                           'scope gate is an observed endpoint, not proof the upstream modality was correct',
                           'frozen gold disagreements require a separate reviewed correction ledger']}
