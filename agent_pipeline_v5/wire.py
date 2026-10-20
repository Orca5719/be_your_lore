"""Compact wire defaults; semantic validation stays in the frozen extractor."""
from copy import deepcopy
from agent_pipeline_v2.extractor import MODALITIES, EVENT_FIELDS, _validate_event
from agent_pipeline_v2_1.extractor import FIELD_ALIASES, REASON_ALIASES

REQUIRED = {'actors', 'event', 'source_ids', 'modality', 'check_reason'}
OPTIONAL = {'explicit', 'mental_state', 'conditions', 'context_ids'}


def _expand(row):
    if not isinstance(row, dict):
        raise ValueError('event must be an object')
    row = deepcopy(row)
    # Same non-semantic aliases as the existing repair wrapper, before defaults.
    for old, new in FIELD_ALIASES.items():
        if old in row and new not in row:
            row[new] = row.pop(old)
    if row.get('check_reason') in REASON_ALIASES:
        row['check_reason'] = REASON_ALIASES[row['check_reason']]
    if not REQUIRED <= set(row) or set(row) - REQUIRED - OPTIONAL:
        raise ValueError('compact event requires actors/event/source_ids/modality/check_reason; no unknown fields')
    if row['modality'] not in MODALITIES:
        raise ValueError('invalid modality')
    expected = row['modality'] != 'inferred'
    if 'explicit' in row and (type(row['explicit']) is not bool or row['explicit'] != expected):
        raise ValueError('explicit conflicts with modality')
    row['explicit'] = expected
    row.setdefault('mental_state', None)
    row.setdefault('conditions', [])
    row.setdefault('context_ids', [])
    return row


def expand_event(row: dict, event_id: str, spans: dict) -> dict:
    event = {'id': event_id, **_expand(row)}
    _validate_event(event, spans, set())
    return event


def expand_payload(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {'events', 'ignored_spans', 'non_event_span_ids'}:
        raise ValueError('invalid three-exit envelope')
    if any(not isinstance(value[key], list) for key in value):
        raise ValueError('three-exit values must be arrays')
    return {**deepcopy(value), 'events': [_expand(e) for e in value['events']]}
